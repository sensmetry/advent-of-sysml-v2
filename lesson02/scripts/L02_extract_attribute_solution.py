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


def show_part_attributes(part: syside.Element) -> None:
    """
    Prints a list of attributes for the input part.
    """

    print(f"Part: {part.name}")
    for owned_element in part.owned_elements:
        if type(owned_element) is syside.AttributeUsage:
            print(f" └ Attribute: {owned_element.name}")


def main() -> None:
    # Load SysML model and get diagnostics (errors/warnings)
    (model, diagnostics) = syside.load_model([MODEL_FILE_PATH])

    # Make sure the model contains no errors before proceeding
    assert not diagnostics.contains_errors(warnings_as_errors=True)

    root_element = find_element_by_name(model, "Reindeer")

    print("\nPrinting part attributes:\n")
    show_part_attributes(root_element)


if __name__ == "__main__":
    main()
