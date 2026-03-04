import pytest
from src.agent.workflow import (
    MultiAgentWorkflow,
    create_workflow,
    WorkflowOutcome,
)
from src.agent.planner import plan_node, PlannerOutcome
from src.agent.validate import validation_node, ValidationOutcome
from src.agent.gen import gen_node


class TestWorkflowOutcome:
    def test_workflow_outcome_creation(self):
        outcome = WorkflowOutcome(
            answer="Test answer",
            source=[{"id": 1, "title": "Test"}],
            reasoning_step=["Step 1"],
            trace_id="test-123",
            langfuse_url="https://langfuse.test/trace/test-123",
        )
        assert outcome.answer == "Test answer"
        assert len(outcome.source) == 1
        assert len(outcome.reasoning_step) == 1
        assert outcome.trace_id == "test-123"


class TestPlannerOutcome:
    def test_planner_outcome_creation(self):
        outcome = PlannerOutcome(
            decision="process",
            route="vector_search",
            reasoning="Test reasoning",
        )
        assert outcome.decision == "process"
        assert outcome.route == "vector_search"
        assert outcome.reasoning == "Test reasoning"

    def test_planner_outcome_optional_route(self):
        outcome = PlannerOutcome(
            decision="direct_answer", reasoning="No processing needed"
        )
        assert outcome.decision == "direct_answer"
        assert outcome.route is None


class TestValidationOutcome:
    def test_validation_outcome_creation(self):
        outcome = ValidationOutcome(
            validation="relevant",
            reasoning="Documents are relevant to the query",
        )
        assert outcome.validation == "relevant"
        assert outcome.reasoning == "Documents are relevant to the query"


class TestWorkflowImports:
    def test_import_workflow_classes(self):
        assert WorkflowOutcome is not None
        assert MultiAgentWorkflow is not None
        assert create_workflow is not None

    def test_import_plan_node(self):
        assert plan_node is not None
        assert PlannerOutcome is not None

    def test_import_validation_node(self):
        assert validation_node is not None
        assert ValidationOutcome is not None

    def test_import_gen_node(self):
        assert gen_node is not None


class TestStateGraphWorkflow:
    def test_workflow_creation(self):
        workflow = create_workflow()
        assert workflow is not None
        assert hasattr(workflow, "app")

    def test_workflow_has_compiled_app(self):
        workflow = create_workflow()
        assert workflow.app is not None

    def test_workflow_has_run_method(self):
        workflow = create_workflow()
        assert hasattr(workflow, "run")
        assert callable(workflow.run)
