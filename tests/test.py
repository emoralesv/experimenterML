from experimenterML.Repetitions import ExpConfig

def main():
    config = ExpConfig.from_yaml("tests/exp_example.yaml")
    config.print()

if __name__ == "__main__":
    main()