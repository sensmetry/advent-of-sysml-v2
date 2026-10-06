import pathlib
import syside

# Path to our SysML model file
LESSON_DIR = pathlib.Path(__file__).parent.parent
MODEL_FILE_PATH = LESSON_DIR / "models" / "L02_SantaSleigh.sysml"


def find_element_by_name(model: syside.Model, name: str) -> syside.Element | None:
    """Search the model for a specific element by name."""

    # Iterates through all model elements that subset Element type
    # e.g. PartUsage, ItemUsage, OccurrenceUsage, etc.
    for element in model.elements(syside.Element, include_subtypes=True):
        if element.name == name:
            return element
    return None


def show_part_decomposition(element: syside.Element, part_level: int = 0) -> None:
    """
    Print parts and all sub-parts in a tree structure.
    Skips attributes, connections, and other non-part elements.
    """

    # Print root element regardless of type
    # e.g. if it is a Package or PartDefinition
    if part_level == 0:
        print(element.name)
    elif type(element) is syside.PartUsage:
        # Indent based on nesting depth
        print("  " * part_level, "└", element.name)

    # Print subparts by calling the same function again for each child
    for owned_element in element.owned_elements.collect():
        show_part_decomposition(owned_element, part_level + 1)


def main() -> None:
    # Load the SysML model; raises syside.ModelError on any error or warning
    model = syside.load_model([MODEL_FILE_PATH], warnings_as_errors=True)

    root_element = find_element_by_name(model, "SantaSleigh")

    print("\nPrinting part decomposition tree:\n")
    show_part_decomposition(root_element)


if __name__ == "__main__":
    main()
