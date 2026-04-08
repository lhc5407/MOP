import subprocess

def run_shell_command_safe(command):
    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            check=False
        )
        return {
            'stdout': result.stdout,
            'stderr': result.stderr,
            'return_code': result.returncode
        }
    except Exception as e:
        return {
            'stdout': '',
            'stderr': str(e),
            'return_code': -1
        }

if __name__ == '__main__':
    test_cmd = 'echo Hello from safe shell command'
    print(run_shell_command_safe(test_cmd))

