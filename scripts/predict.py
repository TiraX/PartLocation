"""
Prediction entry script for Part Location model.

Usage:
    # Single prediction
    python scripts/predict.py --checkpoint outputs/best_model.pth \
        --whole_image data/images/model/model-whole.png \
        --part_image data/images/model/model-part.png

    # Batch prediction
    python scripts/predict.py --checkpoint outputs/best_model.pth \
        --data_dir data/images/ --output results.json
"""

import argparse
import json
import logging
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from part_location.inference.predictor import Predictor


def main():
    parser = argparse.ArgumentParser(description="Part Location Prediction")
    parser.add_argument(
        "--checkpoint", type=str, required=True, help="Path to model checkpoint"
    )
    parser.add_argument(
        "--whole_image", type=str, default=None, help="Path to whole image (single mode)"
    )
    parser.add_argument(
        "--part_image", type=str, default=None, help="Path to part image (single mode)"
    )
    parser.add_argument(
        "--data_dir", type=str, default=None, help="Data directory for batch prediction"
    )
    parser.add_argument(
        "--output", type=str, default=None, help="Output JSON file path (batch mode)"
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.WARNING,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    predictor = Predictor(args.checkpoint)

    if args.whole_image and args.part_image:
        # Single prediction mode
        result = predictor.predict_single(args.whole_image, args.part_image)
        print(json.dumps(result, indent=2))

    elif args.data_dir:
        # Batch prediction mode
        results = predictor.predict_batch(args.data_dir)
        output_str = json.dumps(results, indent=2)

        if args.output:
            with open(args.output, "w", encoding="utf-8") as f:
                f.write(output_str)
            logging.warning("Results saved to %s (%d predictions)", args.output, len(results))
        else:
            print(output_str)

    else:
        parser.error(
            "Provide either (--whole_image + --part_image) or --data_dir"
        )


if __name__ == "__main__":
    main()
