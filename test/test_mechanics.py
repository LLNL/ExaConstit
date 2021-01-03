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
    i = 0
    for ans, test in zip(answers, tests):
        i = i + 1
        for a, t in zip(ans, test):
            err += abs(float(a) - float(t))
    err = err / i
    if (err > 1.0e-10):
        raise ValueError("The following test case failed: ", test_case)
    return True


test_cases = ["voce_pa.toml", "voce_ea.toml", "voce_full.toml", "voce_nl_full.toml",
              "voce_bcc.toml", "voce_full_cyclic.toml", "mtsdd_bcc.toml", "mtsdd_full.toml"]

test_results = ["voce_pa_stress.txt", "voce_ea_stress.txt","voce_full_stress.txt",
                "voce_full_stress.txt", "voce_bcc_stress.txt", "voce_full_cyclic_stress.txt",
                "mtsdd_bcc_stress.txt", "mtsdd_full_stress.txt"]

result = subprocess.run('pwd', stdout=subprocess.PIPE)

pwd = result.stdout.decode('utf-8')

# Remove any stress file that might already be living in the test directory
cmd = 'rm ' + pwd.rstrip() + '/avg_stress.txt'
result = subprocess.run(cmd.rstrip(), stdout=subprocess.PIPE, shell=True)

for test, ans in zip(test_cases, test_results):
    print("Now running test case: " + test)
    cmd = 'mpirun -np 2 ' + pwd.rstrip() + '/../bin/mechanics -opt ' + test
    result = subprocess.run(cmd.rstrip(), stdout=subprocess.PIPE, shell=True)
    ans_pwd = pwd.rstrip() + '/' + ans
    test_pwd = pwd.rstrip() + '/avg_stress.txt'
    check_stress(ans_pwd, test_pwd, test)
    cmd = 'rm ' + pwd.rstrip() + '/avg_stress.txt'
    result = subprocess.run(cmd.rstrip(), stdout=subprocess.PIPE, shell=True)