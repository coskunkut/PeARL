import os
import sys

# get experiment id
if len(sys.argv) == 3:
	batch_id = sys.argv[1]
	switches = sys.argv[2]
elif len(sys.argv) == 2:
	batch_id = sys.argv[1]
	switches = "grasv"
else:
	sys.exit()

if "g" in switches:
	os.system("python3 test/systematic_tests/hasc/main/generator.py {}".format(batch_id))
if "r" in switches:
	os.system("python3 test/systematic_tests/hasc/main/runner.py {}".format(batch_id))
if "a" in switches:
	os.system("python3 test/systematic_tests/hasc/main/analyzer.py {}".format(batch_id))
if "s" in switches:
	os.system("python3 test/systematic_tests/hasc/main/summarizer.py {}".format(batch_id))
if "v" in switches:
	os.system("python3 test/systematic_tests/hasc/main/visualizer.py {}".format(batch_id))
