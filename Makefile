# Specify the target executable
myproject: main.py myproject/*.py
    python3 main.py

# Rule to run the tests
test:
    python3 -m unittest discover -s tests/

# Rule to clean up the build artifacts
clean:
    rm -rf build/ dist/ myproject.egg-info/
    find . -name '*.pyc' -delete
    find . -name '__pycache__' -delete

# Rule to install the project dependencies
install:
    pip3 install -r requirements.txt

# In this example, the Makefile defines four targets: myproject, test, clean, and install. 
# Each target has a set of commands that are executed when the target is invoked.

# The myproject target is the main target of the Makefile, it specifies the executable file 
# and its dependencies and runs the command to execute the project.

# The test target runs the command to test the project using unittest package
# The clean target runs the command to remove the build artifacts, compiled files, 
# and other files that are not needed.
# The install target runs the command to install the project dependencies from the 
# requirements.txt file.
# You can run any of these targets by typing make targetname in the command line, 
# for example make myproject to build the project, or make test to run the tests.


# ADD pyproject.toml