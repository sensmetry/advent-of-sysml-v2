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

    # Iterate through all part definitions and part usages
    for element in model.elements(syside.Type, include_subtypes=True):
        if type(element) in [syside.PartDefinition, syside.PartUsage]:
            element_info = {
                "name": element.name,
                "qualified_name": str(element.qualified_name),
                "type": type(element).__name__,
                "comments": [],
            }

            for comment in element.comments.collect():
                if type(comment) is syside.Comment:
                    element_info["comments"].append({
                        "name": comment.name,
                        "text": comment.body,
                    })

            collected_comments.append(element_info)

    # Sort by qualified name for easier reading
    return sorted(collected_comments, key=lambda x: x["qualified_name"])


def print_to_csv(collected_docs: list[dict], output_file: pathlib.Path) -> None:
    """
    Export documentation to CSV file.
    """
    # Create output directory if it doesn't exist
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Name", "Qualified Name", "Type", "Comment Name", "Comment Text"])

        for item in collected_docs:
            if item["comments"]:
                # If element has docs, write one row per locale
                for comment in item["comments"]:
                    writer.writerow([
                        item["name"],
                        item["qualified_name"],
                        item["type"],
                        comment["name"] or "",
                        comment["text"],
                    ])

    total_comments = sum(len(item["comments"]) for item in collected_docs)
    print(f"\nExported to: {output_file}")
    print(f"Total comment entries: {total_comments}")


def main() -> None:
    # Load the SysML model; raises syside.ModelError on any error or warning
    model = syside.load_model([MODEL_FILE_PATH], warnings_as_errors=True)

    docs = collect_comments(model)
    print_to_csv(docs, CSV_FILE_PATH)


if __name__ == "__main__":
    main()
