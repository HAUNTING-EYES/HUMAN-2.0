from core.version_control import VersionControl

def main():
    vc = VersionControl()
    result = vc.test_version_control()
    print(result)

if __name__ == "__main__":
    main() 