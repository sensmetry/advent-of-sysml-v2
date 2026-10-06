import pathlib
import syside
import csv

# Path to our SysML model file
LESSON_DIR = pathlib.Path(__file__).parent.parent
MODEL_FILE_NAME = "L10_AnnotatedWorkshop"
MODEL_FILE_PATH = LESSON_DIR / "models" / (MODEL_FILE_NAME + ".sysml")
CSV_FILE_PATH = LESSON_DIR / "exports" / (MODEL_FILE_NAME + ".csv")

def collect_documentation(model: syside.Model) -> list[dict]:
    """
    Extract documentation from all parts in the model.
    """
    collected_docs = []

    # Iterate through all part definitions and part usages
    for element in model.elements(syside.Type, include_subtypes=True):
        if type(element) in [syside.PartDefinition, syside.PartUsage]:
            element_info = {
                "name": element.name,
                "qualified_name": str(element.qualified_name),
                "type": type(element).__name__,
                "docs": [],
            }

            for doc in element.documentation.collect():
                element_info["docs"].append({
                    "locale": doc.locale,
                    "text": doc.body,
                })

            collected_docs.append(element_info)

    # Sort by qualified name for easier reading
    return sorted(collected_docs, key=lambda x: x["qualified_name"])


def find_missing_locales(docs: list[dict]) -> None:
    """
    Print elements that are missing documentation in target languages.
    """
    target_locales = ["en_US", "hu_HU", "lt_LT"]

    print("\nReport for missing locales:")
    print("==================================================")

    for element in docs:
        # Get locales this element has documentation for
        element_locales = [doc["locale"] for doc in element["docs"]]

        # Find missing locales
        missing = [loc for loc in target_locales if loc not in element_locales]

        if missing:
            print(f"{element['qualified_name']}")
            print(f"  Missing: {', '.join(missing)}")
            print()

    print("==================================================")


def print_to_csv(collected_docs: list[dict], output_file: pathlib.Path) -> None:
    """
    Export documentation to CSV file.
    """
    # Create output directory if it doesn't exist
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Name", "Qualified Name", "Type", "Locale", "Documentation"])

        for item in collected_docs:
            if item["docs"]:
                # If element has docs, write one row per locale
                for doc in item["docs"]:
                    writer.writerow([
                        item["name"],
                        item["qualified_name"],
                        item["type"],
                        doc["locale"] or "None",
                        doc["text"],
                    ])
            else:
                # If element has no docs, write one row per element
                writer.writerow([
                    item["name"],
                    item["qualified_name"],
                    item["type"],
                    "None",
                    ""
                ])

    total_docs = sum(len(item["docs"]) for item in collected_docs)
    print(f"\nExported to: {output_file}")
    print(f"Total documentation entries: {total_docs}")


def main() -> None:
    # Load the SysML model; raises syside.ModelError on any error or warning
    model = syside.load_model([MODEL_FILE_PATH], warnings_as_errors=True)

    docs = collect_documentation(model)
    find_missing_locales(docs)
    print_to_csv(docs, CSV_FILE_PATH)


if __name__ == "__main__":
    main()
