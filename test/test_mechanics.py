#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import subprocess
import csv
import os
import multiprocessing
import numpy as np
import unittest

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

def runSystemCommands(params):
    test, ans = params
    print("Now running test case: " + test)
    result = subprocess.run('pwd', stdout=subprocess.PIPE)
    pwd = result.stdout.decode('utf-8')
    cmd = 'mpirun -np 2 ' + pwd.rstrip() + '/../bin/mechanics -opt ' + test
    subprocess.run(cmd.rstrip(), stdout=subprocess.PIPE, shell=True)
    ans_pwd = pwd.rstrip() + '/' + ans
    tresult = test.split(".")[0]
    test_pwd = pwd.rstrip() + '/test_'+tresult+'_stress.txt'
    check_stress(ans_pwd, test_pwd, test)
    cmd = 'rm ' + pwd.rstrip() + '/test_'+tresult+'_stress.txt'
    subprocess.run(cmd.rstrip(), stdout=subprocess.PIPE, shell=True)
    return True

def run():
    test_cases = ["voce_pa.toml", "voce_ea.toml", "voce_full.toml", "voce_nl_full.toml",
                "voce_bcc.toml", "voce_full_cyclic.toml", "mtsdd_bcc.toml", "mtsdd_full.toml"]

    test_results = ["voce_pa_stress.txt", "voce_ea_stress.txt","voce_full_stress.txt",
                    "voce_full_stress.txt", "voce_bcc_stress.txt", "voce_full_cyclic_stress.txt",
                    "mtsdd_bcc_stress.txt", "mtsdd_full_stress.txt"]

    result = subprocess.run('pwd', stdout=subprocess.PIPE)

    pwd = result.stdout.decode('utf-8')

    # Remove any stress file that might already be living in the test directory
    for test in test_cases:
        tresult = test.split(".")[0]
        cmd = 'rm ' + pwd.rstrip() + '/test_'+tresult+'_stress.txt'
        result = subprocess.run(cmd.rstrip(), stdout=subprocess.PIPE, shell=True)

    params =  zip(test_cases, test_results)

    # This returns the number of processors we have available to us
    # We divide by 2 since we use 2 cores per MPI call
    # However, this command only works on Unix machines since Windows
    # hasn't added support for this command yet...
    num_processes = int(len(os.sched_getaffinity(0)) / 2)
    # num_processes = multiprocessing.cpu_count() / 2
    print(num_processes)
    pool = multiprocessing.Pool(num_processes)
    pool.map(runSystemCommands, params)
    pool.close()
    pool.join()
    return True

class TestUnits(unittest.TestCase):
    def test_all_cases(self):
        actual = run()
        self.assertTrue(actual)

if __name__ == '__main__':
    unittest.main()