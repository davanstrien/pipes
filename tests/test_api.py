from jobpipe import Plan, embed


def test_embed_without_launch_returns_a_plan(offline):
    plan = embed("org/ds", launch=False, verbose=False)
    assert isinstance(plan, Plan) and plan.task == "embed"
