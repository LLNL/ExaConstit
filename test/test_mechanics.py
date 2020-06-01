#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import subprocess
import csv
import numpy as np

def check_stress(ans_pwd, test_pwd, test_case):
    answers = []
    tests = []
    with open(ans_pwd) as csvfile:
        readcsv = csv.reader(csvfile, delimiter=' ')
        for row in readcsv:
            answers.append(row)
    with open(test_pwd) as csvfile:
        readcsv = csv.reader(csvfile, delimiter=' ')
        for row in readcsv:
            tests.append(row)
    err = 0.0 
    for ans, test in zip(answers, tests):
        for a, t in zip(ans, test):
            err += abs(float(a) - float(t))
    
    if (err > 1.0e-10):
        raise ValueError("The following test case failed: ", test_case)
    return True


test_cases = ["voce_pa.toml", "voce_full.toml", "mtsdd_full.toml"]

test_results = ["voce_pa_stress.txt", "voce_full_stress.txt",
                "mtsdd_full_stress.txt"]

result = subprocess.run('pwd', stdout=subprocess.PIPE)

pwd = result.stdout.decode('utf-8')

for test, ans in zip(test_cases, test_results):
    print("Now running test case: " + test)
    cmd = 'mpirun -np 2 ' + pwd.rstrip() + '/../bin/mechanics -opt ' + test
    result = subprocess.run(cmd.rstrip(), stdout=subprocess.PIPE, shell=True)
    ans_pwd = pwd.rstrip() + '/' + ans
    test_pwd = pwd.rstrip() + '/avg_stress.txt'
    check_stress(ans_pwd, test_pwd, test)
    cmd = 'rm ' + pwd.rstrip() + '/avg_stress.txt'
    result = subprocess.run(cmd.rstrip(), stdout=subprocess.PIPE, shell=True)