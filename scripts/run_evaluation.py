from supportmind.evaluation.runner import run_evaluation


if __name__ == "__main__":
    results = run_evaluation()
    print(results.to_markdown(index=False))
