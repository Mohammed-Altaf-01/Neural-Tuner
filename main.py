from rollout_eval import main as rollout_eval_main


def main() -> int:
    """Default CLI entrypoint: run deterministic evaluation harness."""
    return rollout_eval_main()


if __name__ == "__main__":
    raise SystemExit(main())
