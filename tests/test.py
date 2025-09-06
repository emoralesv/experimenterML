import experimenterML.Experiment as exp
import experimenterML.dashboard as dash
import sys

def main():
    experiment = exp.Experimenter("tests/exp_example.yaml")
    experiment.run()
    
    

if __name__ == "__main__":
    main()