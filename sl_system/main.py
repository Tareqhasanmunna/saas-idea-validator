"""
SaaS Idea Validator - Main Entry Point

Simply run: python main.py
"""

import sys
from ml_system import MLSystem


def main():
    """Execute ML pipeline"""
    try:
        ml_system = MLSystem(config_path='config.yaml')
        ml_system.run_pipeline()

        print("\n" + "="*70)
        print("✓ TRAINING COMPLETE!")
        print("="*70)

        return 0

    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
