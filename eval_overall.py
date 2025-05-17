import os
import subprocess
import json
import signal
import random
random.seed(42)
import shutil
import time
import re
import textwrap
import ast
from pathlib import Path
from tqdm import tqdm
from argparse import ArgumentParser
from copy import deepcopy
from data_utils import read_jsonl
from openai import OpenAI
import google.generativeai as genai
from google.generativeai import GenerationConfig

llm_client = None
failed_generation_count = 0

class TimeoutHandler:
    def __init__(self, timeout, error_message=None):
        self.timeout = timeout
        self.error_message = error_message
    
    def __enter__(self):
        signal.signal(signal.SIGALRM, self.raise_timeout) #SIGALRM only support unix
        signal.alarm(self.timeout)
    
    def __exit__(self, type, value, traceback):
        signal.alarm(0)
    
    def raise_timeout(self, *args):
        raise TimeoutError(self.error_message)
    

def execute(test_code, timeout=5):
    """Try to execute test code and classify the result"""
    try:
        exec_globals = {}
        with TimeoutHandler(timeout):
            exec(test_code, globals())
            return "success"  # No errors
    except AssertionError as e:
        return "assertion_error", e  # Assertion failed
    except TimeoutError:
        return "timeout"  # Timed out
    except Exception as e:
        return "runtime_error", e  # Other runtime errors


def change_function_name(code, new_name):
    """Change the name of the first function in the code to new_name."""
    try:
        # Parse the code into an AST
        tree = ast.parse(code)

        # Find the first function definition and change its name
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                if node.name != new_name:
                    node.name = new_name
                    break
                else:
                    break

        # Convert the modified AST back to code
        new_code = ast.unparse(tree)
        return new_code
    except Exception as e:  # cannot parse
        return code


def remove_extra(testcase, func_name, lang='python'):
    """Remove extra test inputs and natural language descriptions before and after the test method.
    Only keep the contents between def test() and solution.{func_name}"""
    lines=testcase.split('\n')
    func_startline=0 #the line when test function starts (def test....)
    for i in range(len(lines)):
        if lines[i].find('def test')>=0:
            func_startline=i
            break
    test_endline=len(lines)
    for i in range(len(lines)):
        if lines[i].find(f'solution.{func_name}')>=0: #first call to the function under test
            test_endline=i+1
            break
    new_testcase='\n'.join(lines[func_startline:test_endline])
    return new_testcase


def reformat_case_byrules(testcase, func_name, lang='python'):
    """Reformat a test case by removing indents and changing function name if needed."""
    if testcase.startswith(' '):  # remove extra indents
        testcase = textwrap.dedent(testcase)
    lines = testcase.split('\n')

    if lang == 'python':
        last_line = lines[-1]  # if last line is not complete (due to token limit), remove it
        last_line = textwrap.dedent(last_line)
        try:
            compile(last_line, '<string>', 'exec')
        except:
            lines = lines[:-1]  # last line cannot compile

    testcase = '\n'.join(lines)
    testcase = change_function_name(testcase, func_name)
    return testcase


def regenerate_testcase(task_num, func_name, code, j, current_testcase, res, args, iteration_num):
    """
    Regenerate a test case using the specified OpenAI model when the current test case fails.
    
    Args:
        task_num: The task number/ID
        func_name: The name of the function under test
        code: The code being tested
        j: The test index
        current_testcase: The current failing test case
        res: Excution result
        args: Command line arguments containing model configuration
        iteration_num: Nth iteration of regeneration
    
    Returns:
        A reformatted test case
    """
    
    global failed_generation_count, llm_client
    
    # Create system prompt for test generation
    system_prompt = """You are an expert Python programmer specializing in test case generation. 
When provided with code and a failing test, generate a new test case that avoids the error.
Respond ONLY with Python code for the test case - no explanations, comments, or markdown."""
    
    # Format the error info for the prompt
    error_description = str(res)
    if isinstance(res, tuple):
        error_description = f"Error type: {res[0]}, Message: {str(res[1])}"
    
    # Create user prompt with all required information
    user_prompt = f"""The following Python test case failed:

```python
{current_testcase}
```

The error was: {error_description}

This test was trying to test the following code:

```python
{code}
```

Generate a new test case that will work correctly. The test function should be named 'test_{func_name}' 
and should begin with 'solution = Solution()'. Only include the test code, no explanations."""
    
    # Call LLM Client API
    try:
        generated_test = ''
        if args.client == 'openAI':
            response = llm_client.chat.completions.create(
                model=args.regen_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=args.regen_temperature,
                max_tokens=args.regen_max_tokens
            )
            
            # Extract the generated test case
            generated_test = response.choices[0].message.content.strip()
        else:
            final_prompt = system_prompt + '\n' + user_prompt
            generation_config = GenerationConfig(
                candidate_count=1,
                max_output_tokens=args.regen_max_tokens,
                temperature=args.regen_temperature
            )
            generated=llm_client.generate_content(final_prompt, generation_config=generation_config)
            if generated.candidates[0].finish_reason==1: #normal stop
                generated_test=generated.text
            else:
                generated_test=''
                failed_generation_count+=1
        
        # Remove markdown code blocks if present
        if generated_test.startswith("```python"):
            generated_test = generated_test.replace("```python", "").replace("```", "").strip()
        elif generated_test.startswith("```"):
            generated_test = generated_test.replace("```", "").strip()
        
        # Process and reformat the test case
        test_funcname = f'test_{func_name}'
        extracted_testcase = remove_extra(generated_test, func_name)
        reformatted_testcase = reformat_case_byrules(extracted_testcase, test_funcname, 'python')
        
        # Save the regenerated test case to a JSONL file
        output_dir = Path('predictions')
        output_dir.mkdir(exist_ok=True)
        output_file = output_dir / f"regenerated_tests_{args.regen_model}_iteration_num{iteration_num}.jsonl"

        # Create entry to save
        test_entry = {
            "task_num": task_num,
            "func_name": func_name,
            "test_index": j,
            "code": code,
            "orginial_test": current_testcase,
            "error": error_description,
            "regenerated_test": reformatted_testcase,
        }
        
        # Append to file if exists, create otherwise
        if output_file.exists():
            with open(output_file, 'a') as f:
                f.write(json.dumps(test_entry) + '\n')
        else:
            with open(output_file, 'w') as f:
                f.write(json.dumps(test_entry) + '\n')
        
        return reformatted_testcase
        
    except Exception as e:
        print(f"Error calling LLM API: {e}")
        failed_generation_count+=1
        # Return the original test case if regeneration fails
        return current_testcase
        
    
def coverage_at_k_sample(passed_tests, k, cov_command_prefix):
    """Compute coverage@k for a single program under test."""
    random.shuffle(passed_tests)
    if len(passed_tests)>=k:
        #num_splits=math.ceil(len(passed_tests)/k) #round up or down?
        num_splits=len(passed_tests)//k
        splited_tests=[passed_tests[i * k : (i + 1) * k] for i in range(num_splits)]
    else: #if number of passed tests is less than k, do not split
        splited_tests=[passed_tests]
    #calculate and average coverages for each group
    split_line_covs=[]
    split_branch_covs=[]
    
    for i,test_group in enumerate(splited_tests):
        group_line_cov=[]
        group_branch_cov=[]
        cov_command=deepcopy(cov_command_prefix)
        for test in test_group:
            cov_command.append(test)
            subprocess.run(cov_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            cov_report=json.load(open('coverage.json'))
            total_stmt=cov_report['totals']['num_statements']
            covered_stmt=cov_report['totals']['covered_lines']
            line_cov=covered_stmt/total_stmt
            total_branch=cov_report['totals']['num_branches']
            covered_branch=cov_report['totals']['covered_branches']
            branch_cov=covered_branch/total_branch
            group_line_cov.append(line_cov)
            group_branch_cov.append(branch_cov)
        
        group_avg_line_cov=sum(group_line_cov)/len(group_line_cov)
        group_avg_branch_cov=sum(group_branch_cov)/len(group_branch_cov)
        split_line_covs.append(group_avg_line_cov)
        split_branch_covs.append(group_avg_branch_cov)

    avg_line_cov=sum(split_line_covs)/len(split_line_covs)
    avg_branch_cov=sum(split_branch_covs)/len(split_branch_covs)
    return {'line_cov':avg_line_cov,'branch_cov':avg_branch_cov}
        
    

def check_correctness(generated_data, args, ks=[1, 2, 5]):
    """Compute syntactical and execution correctness (with coverage)."""
    total_cases=0
    total_syn_correct=0
    total_comp_correct=0 
    total_assertion_correct = 0
    total_exec_correct=0
    syn_failed=0

    exec_fails=[]

    total_line_cov=0
    total_branch_cov=0
    line_covs_at_k={f'cov@{k}':[] for k in ks}
    branch_covs_at_k={f'cov@{k}':[] for k in ks}

    remove_pattern=re.compile(r'tmp*')

    for i, data in tqdm(enumerate(generated_data)):
        task_num=data['task_num']
        difficulty=data['difficulty']
        func_name=data['func_name']
        code=data['code']
        test_cases=data['tests']
        test_import=f'from tmp_{i}_{difficulty}_{args.client}.under_test import Solution\n'
        test_import_simple=f'from under_test import Solution\n'
        os.makedirs(f'tmp_{i}_{difficulty}_{args.client}',exist_ok=True) #create different tmp folders for different problems to avoid conflicts
        with open(f'tmp_{i}_{difficulty}_{args.client}/under_test.py','w') as f: #write program under test and test cases into tmp files
            f.write(code)
        passed_tests=[]

        for j, testcase in enumerate(test_cases):
            #testcase=textwrap.dedent(testcase)
            total_cases+=1
            try:
                res=compile(testcase,'<string>','exec') #check syntax correctness
                total_syn_correct+=1

                test_code=test_import+testcase+f'\ntest_{func_name}()'
                time.sleep(0.01)
                res=execute(test_code)
                if res == "success":
                    if test_code.find(f'solution.{func_name}')==-1: #if the function under test is not called, also consider as failed
                        print('func under test not called')
                        exec_fails.append({'task':task_num,'test_num':j,'error':'not called'})
                    else:
                        total_exec_correct+=1
                        total_assertion_correct += 1
                        test_code_simple=test_import_simple+testcase #write to files for computing coverage
                        with open(f'tmp_{i}_{difficulty}_{args.client}/test_{j}.py','w') as f:
                            f.write(test_code_simple)
                        passed_tests.append(f'test_{j}.py')
                else:
                    if args.run_debugger == 'false':
                        if isinstance(res, tuple) and res[0] == "assertion_error":
                            total_exec_correct += 1
                        else:
                            exec_fails.append({'task':task_num,'test_num':j,'error':res})
                    else:
                        # Try to regenerate the test case up to N times for both assertion errors and other errors
                        max_regenerations = args.regen_max_iteration
                        regeneration_count = 0
                        current_testcase = testcase

                        while regeneration_count < max_regenerations:
                            # Call regenerate_testcase function to get a new test case
                            new_testcase = regenerate_testcase(task_num, func_name, code, j, current_testcase, res, args, regeneration_count+1)
                            current_testcase = new_testcase

                            # Try to execute the new test case
                            test_code = test_import + current_testcase + f'\ntest_{func_name}()'
                            time.sleep(0.01)
                            res = execute(test_code)

                            if res == "success":
                                if test_code.find(f'solution.{func_name}') == -1:
                                    print('func under test not called')
                                    exec_fails.append({'task':task_num,'test_num':j,'error':'not called'})
                                    break
                                else:
                                    total_exec_correct += 1
                                    total_assertion_correct += 1
                                    test_code_simple = test_import_simple + current_testcase
                                    with open(f'tmp_{i}_{difficulty}_{args.client}/test_{j}.py', 'w') as f:
                                        f.write(test_code_simple)
                                    passed_tests.append(f'test_{j}.py')
                                    break

                            regeneration_count += 1

                        if regeneration_count == max_regenerations:
                            if isinstance(res, tuple) and res[0] == "assertion_error":
                                total_exec_correct += 1
                            else:
                                exec_fails.append({'task':task_num, 'test_num':j, 'error': f'Failed after {max_regenerations} regenerations with the last error: {res}'})

            except:
                syn_failed+=1
                #print('syntax error')
                #print(testcase)
                pass

        if len(passed_tests)>0: #start measuring coverage
            #total coverage for all tests
            cov_command_prefix=['pytest', '--cov=under_test', '--cov-branch', '--cov-report=json:coverage.json']
            subprocess.run(f'cp .coveragerc tmp_{i}_{difficulty}_{args.client}/.coveragerc',shell=True) #copy config file to tmp_folder
            os.chdir(f'tmp_{i}_{difficulty}_{args.client}') #enter tmp_ folder for testing
            cov_command=deepcopy(cov_command_prefix)
            for test in passed_tests:
                cov_command.append(test)

            try:
                subprocess.run(cov_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                cov_report=json.load(open('coverage.json'))
                total_stmt=cov_report['totals']['num_statements']
                covered_stmt=cov_report['totals']['covered_lines']
                line_cov=covered_stmt/total_stmt
                total_branch=cov_report['totals']['num_branches']
                covered_branch=cov_report['totals']['covered_branches']
                branch_cov=covered_branch/total_branch
                total_line_cov+=line_cov
                total_branch_cov+=branch_cov
                #print(f'Line Coverage: {line_cov}, Branch Coverage: {branch_cov}')
            except: #unknown pytest error: cannot generate coverage report (AssertionError: Expected current collector to be <Collector at 0x7f7d2db07810: CTracer>, but it's <Collector at 0x7f7d2cd794d0: CTracer>)
                print('Failed to generate coverage report')
                pass

            #compute coverage@k
            for k in ks:
                res_at_k=coverage_at_k_sample(passed_tests,k,cov_command_prefix)
                line_covs_at_k[f'cov@{k}'].append(res_at_k['line_cov'])
                branch_covs_at_k[f'cov@{k}'].append(res_at_k['branch_cov'])

            os.chdir('..') #exit tmp_ folder
        else: #no test cases passed
            pass

    for dirpath, dirnames, filenames in os.walk('./', topdown=False): #execute() runs too fast, remove dirs at last
        # Filter dirnames based on the regex pattern
        for dirname in dirnames:
            if remove_pattern.match(dirname):
                shutil.rmtree(dirname)

    syn_correct=total_syn_correct/total_cases
    exec_correct=total_exec_correct/total_cases
    assertion_correct = total_assertion_correct / total_cases
    print(f'Syntax Correctness: {syn_correct}')
    print(f'Assertion Correctness: {assertion_correct}')
    print(f'Executable Correctness: {exec_correct}')

    # Write results to results/correctness_results.jsonl in JSONL format
    os.makedirs('results', exist_ok=True)
    results = {
        "Syntax Correctness": syn_correct,
        "Assertion Correctness": assertion_correct,
        "Executable Correctness": exec_correct
    }
    with open(f'results/correctness_results_{args.client}_{args.run_debugger}.jsonl', 'w') as f:
        f.write(json.dumps(results) + '\n')

    #compute average coverage@k
    coverage_results = {}
    for k in ks:
        line_covs_at_k[f'cov@{k}']=sum(line_covs_at_k[f'cov@{k}'])/len(generated_data)
        branch_covs_at_k[f'cov@{k}']=sum(branch_covs_at_k[f'cov@{k}'])/len(generated_data)
        print(f'line coverage@{k}',line_covs_at_k[f'cov@{k}'])
        print(f'branch coverage@{k}',branch_covs_at_k[f'cov@{k}'])
        coverage_results[f'line_coverage@{k}'] = line_covs_at_k[f'cov@{k}']
        coverage_results[f'branch_coverage@{k}'] = branch_covs_at_k[f'cov@{k}']

    

    #compute coverage
    avg_line_cov=total_line_cov/len(generated_data)
    avg_branch_cov=total_branch_cov/len(generated_data)
    print(f'Average Line Coverage: {avg_line_cov}, Average Branch Coverage: {avg_branch_cov}')
    coverage_results['avg_line_cov'] = avg_line_cov
    coverage_results['avg_branch_cov'] = avg_branch_cov

    # Write coverage@k results to results/coverage@k_results.jsonl in JSONL format
    with open(f'results/coverage@k_results_{args.client}_{args.run_debugger}.jsonl', 'w') as f:
        f.write(json.dumps(coverage_results) + '\n')

    # Write execution failures to results/execution_fails.jsonl in JSONL format
    with open(f'results/execution_fails_{args.client}_{args.run_debugger}.jsonl', 'w') as f:
        for fail in exec_fails:
            # Convert non-JSON-serializable objects (like IndexError) to strings
            fail_copy = fail.copy()
            if isinstance(fail_copy.get('error'), tuple) and len(fail_copy['error']) > 1 and isinstance(fail_copy['error'][1], Exception):
                fail_copy['error'] = (fail_copy['error'][0], str(fail_copy['error'][1]))
            elif isinstance(fail_copy.get('error'), Exception):
                fail_copy['error'] = str(fail_copy['error'])
            f.write(json.dumps(fail_copy) + '\n')

    print(f'Failed generation count: {failed_generation_count}')
    return {'syn_correct':syn_correct,'exec_correct':exec_correct}, exec_fails


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--path", type=str, default='totalcov_gpt-3.5-turbo.jsonl')
    parser.add_argument("--ks", type=int, nargs='+', default=[1, 2, 5])
    # LLM parameters for test case regeneration
    parser.add_argument("--client", type=str, default='openAI',
                        choices=['openAI', 'gemini'])
    parser.add_argument("--regen_model", type=str, default='gpt-4o-mini', 
                        choices=['gpt-3.5-turbo', 'gpt-4', 'gpt-4-turbo', 'gpt-4o', 'gpt-4o-mini', 'gemini-2.0-flash'],
                        help='The OpenAI model to use for test regeneration')
    parser.add_argument("--regen_temperature", type=float, default=0,
                        help='Temperature for test regeneration')
    parser.add_argument("--regen_max_tokens", type=int, default=256,
                        help='Maximum number of tokens for test regeneration')
    parser.add_argument("--regen_max_iteration", type=int, default=3,
                        help="Maximum number of iterations for test regeneration for failed tests")
    parser.add_argument("--run_debugger", type=str, default='false',
                        choices=['true', 'false'],
                        help="Should run debugger or not")
    return parser.parse_args()


if __name__=='__main__':
    args=parse_args()
    print(f"Evaluation file: {args.path}")
    print(f"Coverage k values: {args.ks}")
    print(f"LLM client: {args.client}")
    print(f"Regeneration model: {args.regen_model}")
    print(f"Regeneration temperature: {args.regen_temperature}")
    print(f"Regeneration max tokens: {args.regen_max_tokens}")
    print(f"Regeneration max iteration: {args.regen_max_iteration}")
    print(f"Run debugger: {args.run_debugger}")

    if args.run_debugger == 'true':
        if args.client == 'openAI':
            # Initialize OpenAI client once
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                print("Warning: OPENAI_API_KEY environment variable not set. Test regeneration will not work.")
            else:
                llm_client = OpenAI(api_key=api_key)
                print("OpenAI client initialized successfully")
        elif args.client == 'gemini':
            api_key=os.getenv("GOOGLE_API_KEY")
            if not api_key:
                print("Warning: GOOLE_API_KEY environment variable not set. Test regeneration will not work.")
            else:
                genai.configure(api_key=api_key)
                llm_client = genai.GenerativeModel(args.regen_model)
                print("Gemini client initialized successfully")
        else:
            print("Warning: not a valid LLM client")

    
    output_dir = Path('predictions')
    predictions=read_jsonl(output_dir / args.path)
    print(f"Total predictions: {len(predictions)}")
    
    check_correctness(predictions, args, ks=args.ks)