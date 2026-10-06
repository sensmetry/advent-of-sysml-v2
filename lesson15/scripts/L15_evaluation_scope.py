import pathlib
import syside

# Path to our SysML model file
LESSON_DIR = pathlib.Path(__file__).parent.parent
MODEL_FILE_PATH = LESSON_DIR / "models" / "L15_BasicFeatureEvaluation.sysml"
CONTEXT = syside.vm.Context(syside.Environment.get_default().lib)


def find_element_by_name(model: syside.Model, name: str) -> syside.Element | None:
    """Search the model for a specific element by name."""

    for element in model.elements(syside.Element, include_subtypes=True):
        if element.name == name:
            return element
    return None


def evaluate_feature(
    feature: syside.Feature, scope: syside.Type
) -> syside.Value | None:
    compiler = syside.Compiler()
    value, compilation_report = compiler.evaluate_feature(
        feature=feature,
        scope=scope,
        context=CONTEXT,
        experimental_quantities=True,
    )
    if compilation_report.fatal:
        print(compilation_report.diagnostics)
        exit(1)
    return value


def main() -> None:
    # Load the SysML model; raises syside.ModelError on any error or warning
    model = syside.load_model([MODEL_FILE_PATH], warnings_as_errors=True)

    # Find the SantaSleigh element in the model
    reindeer = find_element_by_name(model, "Reindeer")
    rudolph = find_element_by_name(model, "Rudolph")

    weight_reindeer = evaluate_feature(reindeer["weight"], reindeer)
    weight_rudolph = evaluate_feature(reindeer["weight"], rudolph)

    print(f"Rudolph weight: {weight_rudolph}")
    print(f"Reindeer weight: {weight_reindeer}")


if __name__ == "__main__":
    main()
