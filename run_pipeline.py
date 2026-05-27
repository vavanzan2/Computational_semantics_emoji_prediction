# =============================================================================
# run_pipeline.py — Entry point for the full pipeline
#
# Run this file to execute all steps in sequence:
#
#   python run_pipeline.py
#
# Pipeline structure:
#   Step 1 → Loads the Excel spreadsheet and prepares the data
#   Step 2 → Identifies experimental pairs and extracts inserted emojis
#   Step 3 → Predicts emojis in artificial pairs (with and without context)
#   Step 4 → Predicts emojis in spontaneous turns (with and without context)
#   Step 5 → Generates the statistical analysis and saves the plots
#
# Intermediate files are saved in data/
# Final plots are saved in plots/
# =============================================================================

import step1_load_and_prepare
import step2_build_pairs
import step3_predict_artificial
import step4_predict_spontaneous
import step5_analysis


def main():
    print("=" * 60)
    print("   PIPELINE: Effect of context on emoji prediction ")
    print("=" * 60)

    step1_load_and_prepare.run()
    step2_build_pairs.run()
    step3_predict_artificial.run()
    step4_predict_spontaneous.run()
    step5_analysis.run()

    print("\n" + "=" * 60)
    print("  Pipeline conclude successfuly.")
    print("=" * 60)


if __name__ == "__main__":
    main()
