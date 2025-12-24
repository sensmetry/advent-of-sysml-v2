import pathlib
import syside

## ============================================================================
## CONFIGURATION
## ============================================================================
LESSON_DIR = pathlib.Path(__file__).parent.parent
MODEL_FILE_PATH = LESSON_DIR / "models" / "L24_RequirementSatisfaction.sysml" 
STANDARD_LIBRARY = syside.Environment.get_default().lib

## ============================================================================
## SYSML MODEL PARSING
## ============================================================================
class ModelParsing:
    @staticmethod
    def find_element_by_name(
        model: syside.Model,
        name: str,
    ) -> syside.Element | None:
        """Search the model for a specific element by name."""

        for element in model.elements(syside.Element, include_subtypes=True):
            if element.name == name:
                return element
        return None
    
    @staticmethod
    def find_owned_elements_by_type(
        parent: syside.Element,
        searchType: syside.Type,
    ) -> list[syside.Element]:
        """Search the model for a specific elements by type."""
        result = []
        for element in parent.owned_elements:
            if type(element) is searchType:
                result.append(element)
        return result

    @staticmethod
    def find_inherited_features_by_type(
        parent: syside.Element,
        searchType: syside.Type,
        excludeLibrary: bool = True,
    ) -> list[syside.Feature]:
        """Search for inherited features by type."""
        result = []
        for feature in parent.inherited_features:
            if type(feature) is searchType:
                result.append(feature)

        # Filter out library features
        if excludeLibrary:
            result = [x for x in result if x.is_library_element is False]

        return result

    @staticmethod
    def find_owned_elements_by_name(
        parent: syside.Element,
        name: str,
    ) -> syside.Element | None:
        """Search for a specific owned element by name."""

        for element in parent.owned_elements:
            if element.name == name:
                return element
        return None

    @staticmethod
    def evaluate_feature(
        feature: syside.Feature,
        scope: syside.Type,
    ) -> syside.Value | None:
        """Evaluate a feature within a given scope and return its computed value."""
        compiler = syside.Compiler()
        value, compilation_report = compiler.evaluate_feature(
            feature=feature,
            scope=scope,
            stdlib=STANDARD_LIBRARY,
            experimental_quantities=True,
        )
        if compilation_report.fatal:
            print(compilation_report.diagnostics)
            exit(1)
        return value

    @staticmethod
    def evaluate_constraint(
        constraint: syside.ConstraintUsage,
        subject: syside.Element,
    ) -> bool | None:
        """Evaluate ConstraintUsage within a given subject scope."""
        result = None

        if constraint.result_expression is not None:
            result = ModelParsing.evaluate_feature(
                constraint.result_expression, subject
            )

        return result

    @staticmethod
    def get_element_docs(element: syside.Element) -> list[syside.Documentation]:
        """Get all element member documentations."""

        return [
            x.body.replace("\n", "")
            for x in element.members
            if type(x) is syside.Documentation and x.is_library_element is False
        ]

    @staticmethod
    def get_inherited_name(element: syside.Element) -> str:
        """Return element name or inherited name if it doesn't exist."""
        short_name = element.short_name
        name  = element.name

        if short_name is None and name is None:
            specializations = [
                x for x in element.heritage.elements if x.is_library_element is False
            ]
            for specialization in specializations:
                if specialization.short_name is not None or specialization.name is not None:
                    short_name = specialization.short_name
                    name = specialization.name
                    break

        if short_name and name:
            return "<" + short_name + "> " + name
        elif short_name:
            return short_name
        else:
            return name

## ============================================================================
## VERIFICATION
## ============================================================================
class Verification:
    def verify_requirement(
        requirement: syside.RequirementUsage,
        scope: syside.Element,
        level: int = 0,
    ) -> bool:
        """Verify requirement and its subrequirements by asserting that the constraints are met."""
        result = True

        docs = ModelParsing.get_element_docs(requirement)
        name = ModelParsing.get_inherited_name(requirement)
        print("   " * level, f"└ REQ [{name}]: {docs[0] if len(docs) > 0 else None}")

        # Get subject from requirement and evaluate its value within context
        subjects = [x for x in requirement.memberships if type(x) is syside.SubjectMembership]
        assert len(subjects) == 1
        subject_value = ModelParsing.evaluate_feature(subjects[0].targets[0], scope)

        # Get assume and require constraints by looking at owningMembership (parent) kind
        # This is better illustrated by CST structure of `<kind> constraint`:
        # -------------------------------------------------------------------------------
        # children: RequirementConstraintMembership     <-- constraint.parent
        #   kind: RequirementConstraintKind             <-- constraint.parent.kind
        #   target: ConstraintUsage                     <-- constraint
        constraints = ModelParsing.find_inherited_features_by_type(requirement, syside.ConstraintUsage)
        assume_constraints = [x for x in constraints if x.parent.kind.name == 'Assumption']
        require_constraints = [x for x in constraints if x.parent.kind.name == 'Requirement']

        assume_results = [ModelParsing.evaluate_constraint(assume, subject_value) for assume in assume_constraints]

        # Proceed only if all assume constraints have been met
        if False not in assume_results:
            require_results = [ModelParsing.evaluate_constraint(require, subject_value) for require in require_constraints]
            result = False not in require_results

        # Recursively verify all child requirements
        inherited_reqs = ModelParsing.find_inherited_features_by_type(requirement, syside.RequirementUsage)
        for sub_req in inherited_reqs:
            result &= Verification.verify_requirement(sub_req, scope, level + 1)

        print("   " * (level+1), f"└ Requirement satisfied: {"✅" if result else "❌"}")

        return result


    def verify_requirements(element : syside.Element) -> bool:
        """Verify all owned requirements of input element."""
        all_tests_pass = True

        satisfy_requirements = ModelParsing.find_owned_elements_by_type(
            element, syside.SatisfyRequirementUsage
        )
        print("=" * 100)

        for req in satisfy_requirements:
            print(f"[{element.name}]")
            all_tests_pass &= Verification.verify_requirement(req, element)

        print(f"\nAll [{element.name}] requirements satisfied: {"✅" if all_tests_pass else "❌"}")

        return all_tests_pass


def main() -> None:
    # Load SysML model and get diagnostics (errors/warnings)
    (model, diagnostics) = syside.load_model([MODEL_FILE_PATH])

    # Make sure the model contains no errors before proceeding
    assert not diagnostics.contains_errors(warnings_as_errors=True)

    # Find elements and evaluate their requirements
    ordinary_cargo_bay = ModelParsing.find_element_by_name(model, "OrdinaryCargoBay")
    Verification.verify_requirements(ordinary_cargo_bay)

    magical_cargo_bay = ModelParsing.find_element_by_name(model, "MagicalCargoBay")
    Verification.verify_requirements(magical_cargo_bay)

    santa_sleigh_design = ModelParsing.find_element_by_name(model, "SantaSleighDesign")
    Verification.verify_requirements(santa_sleigh_design["SantaSleigh"])
    
    print("=" * 100)

if __name__ == "__main__":
    main()
