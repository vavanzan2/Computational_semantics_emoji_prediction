# =============================================================================
# # main.py — Controls the pipeline through the command line
#
# Usage examples:
#
#   Run all steps:
#       python main.py --all
#
#   Run a specific step:
#       python main.py --step 1
#       python main.py --step 3
#
#   Run a range of steps:
#       python main.py --step 2 --to 4
#
#   Show help:
#       python main.py --help
# =============================================================================

import argparse
import sys

import step1_load_and_prepare
import step2_build_pairs
import step3_predict_artificial
import step4_predict_spontaneous
import step5_analysis

STEPS = {
    1: ("Load and prepare data",              step1_load_and_prepare.run),
    2: ("Create experimental pairs",          step2_build_pairs.run),
    3: ("Predict artifical emojis",            step3_predict_artificial.run),
    4: ("Predict spontaneous emojis ",            step4_predict_spontaneous.run),
    5: ("Analysis ",         step5_analysis.run),
}


def list_steps():
    print("\nSteps available:")
    for number, (description, _) in STEPS.items():
        print(f"  {number} — {description}")
    print()


def execute_steps(numbers):
    for number in numbers:
        description, function = STEPS[number]
        print(f"\n{'─' * 60}")
        print(f"  Executing Step {number}: {description}")
        print(f"{'─' * 60}")
        function()

    print(f"\n{'=' * 60}")
    print("  Conclude.")
    print(f"{'=' * 60}\n")


def main():
    parser = argparse.ArgumentParser(
        prog="main.py",
        description="Pipeline: Effect of context on emoji prediction ",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python main.py --all\n"
            "  python main.py --step 1\n"
            "  python main.py --step 2 --ate 4\n"
            "  python main.py --list"
        )
    )

    parser.add_argument(
        "--all",
        action="store_true",
        help="Execute all steps in sequence (1 → 5)"
    )
    parser.add_argument(
        "--step",
        type=int,
        choices=list(STEPS.keys()),
        metavar="N",
        help=f"Step to execute (1–{max(STEPS)})"
    )
    parser.add_argument(
        "--to",
        type=int,
        choices=list(STEPS.keys()),
        metavar="N",
        help=(
            f"Used with --step: execute from step N to step informed here.\n"
            f"Example: --step 2 --to 4  execute steps 2, 3 e 4"
        )
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List all steps available and conclude"
    )

    args = parser.parse_args()

    # No arguments → show help
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(0)

    if args.list:
        list_steps()
        sys.exit(0)

    if args.all:
        execute_steps(list(STEPS.keys()))
        return

    if args.step:
        begin = args.step
        end    = args.ate if args.ate else args.step

        if end < begin:
            parser.error(
                f"--up to ({end}) cannot be lower than --step ({begin})."
            )

        numeros = list(range(begin, end + 1))
        execute_steps(numeros)
        return

    parser.print_help()


if __name__ == "__main__":
    main()
