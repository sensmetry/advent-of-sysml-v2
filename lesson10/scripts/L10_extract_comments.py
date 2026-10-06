import pathlib
import syside
import csv

# Path to our SysML model file
LESSON_DIR = pathlib.Path(__file__).parent.parent
MODEL_FILE_NAME = "L10_AnnotatedWorkshop"
MODEL_FILE_PATH = LESSON_DIR / "models" / (MODEL_FILE_NAME + ".sysml")
CSV_FILE_PATH = LESSON_DIR / "exports" / (MODEL_FILE_NAME + ".csv")

def collect_comments(model: syside.Model) -> list[dict]:
    """
    Extract comments from all parts in the model.
    """
    collected_comments = []

    # TODO: Implement comment extraction

    return collected_comments


def print_to_csv(collected_docs: list[dict], output_file: pathlib.Path) -> None:
    """
    Export documentation to CSV file.
    """
    
    # TODO: Implement printing to CSV


def main() -> None:
    # Load the SysML model; raises syside.ModelError on any error or warning
    model = syside.load_model([MODEL_FILE_PATH], warnings_as_errors=True)

    docs = collect_comments(model)
    print_to_csv(docs, CSV_FILE_PATH)


if __name__ == "__main__":
    main()
