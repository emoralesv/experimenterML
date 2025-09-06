from experimenterML.Repetitions import Repetitions

def main():
    reps = Repetitions("tests/exp_example.yaml")
    while reps.next() is not None:
        reps.print(current=True)
        reps.realize()


            

        
        #if os.path.exists(model_path) or config["exp_name"] in done_exps:
         #   print(f"✔️  Saltando {config['exp_name']} (ya realizado)")
          #  continue

        result = run_experiment(config, device)
        #if result is not None:
         #   df_results = pd.concat([df_results, pd.DataFrame([result])], ignore_index=True)
          #  df_results.to_csv(results_path, index=False)

if __name__ == "__main__":
    main()