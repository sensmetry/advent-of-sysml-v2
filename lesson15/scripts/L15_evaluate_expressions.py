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


def evaluate_expression(expression: syside.Expression) -> syside.Value | None:
    compiler = syside.Compiler()
    value, compilation_report = compiler.evaluate(
        expr=expression,
        context=CONTEXT,
        experimental_quantities=True,
    )
    if compilation_report.fatal:
        print(f"{expression} -> {expression.arguments.collect()}")
        print(compilation_report.diagnostics)
        exit(1)
    return value


def evaluate_attribute_values(element: syside.Element) -> list[dict]:
    """
    TBD
    """

    results = dict()

    for attribute in element.owned_elements.collect():
        if isinstance(attribute, syside.Feature):
            results[attribute.name] = evaluate_expression(
                attribute.feature_value_expression
            )

    return results


def main() -> None:
    # Load the SysML model; raises syside.ModelError on any error or warning
    model = syside.load_model([MODEL_FILE_PATH], warnings_as_errors=True)

    # Find the Rudolph element in the model
    rudolph_element = find_element_by_name(model, "Rudolph")

    # Evaluate Rudolph's attributes
    rudolph_values = evaluate_attribute_values(rudolph_element)
    print(f"Rudolph values: {rudolph_values}")

    # For simple values like weight and energyLevel, we get direct results
    weight = rudolph_values["weight"]
    energy = rudolph_values["energyLevel"]
    print(f" - Weight: {weight}\n - Energy: {energy}")

    # For complex attributes like 'features', we get a syside object
    # We need to evaluate that object to access its attributes
    rudolph_feature_values = evaluate_attribute_values(rudolph_values["features"])
    print(f"Feature values: {rudolph_feature_values}")

    nose_color = rudolph_feature_values["noseColor"]
    eye_color = rudolph_feature_values["eyeColor"]
    antler_length = rudolph_feature_values["antlerLength"]
    print(f" - Nose: {nose_color}\n - Eyes: {eye_color}\n - Antlers: {antler_length}")


if __name__ == "__main__":
    main()
