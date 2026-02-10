import sys
import subprocess
import pkgutil

def main():
    if pkgutil.find_loader('shap'):
        print('shap-ok')
        return
    print('installing shap...')
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '--user', 'shap'])
    print('installed')

if __name__ == '__main__':
    main()
