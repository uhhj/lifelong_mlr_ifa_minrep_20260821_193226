"""A new explicit order; original benchmark order 0 remains intact."""
ORIGINAL_TASKS=[0,1,2,3,4,5,9,8,7,6]
ORDER_INDEX=21
def install():
    from libero.libero import benchmark
    assert benchmark.task_orders[0]==list(range(10))
    if len(benchmark.task_orders)==ORDER_INDEX:
        benchmark.task_orders.append(list(ORIGINAL_TASKS))
    assert benchmark.task_orders[ORDER_INDEX]==ORIGINAL_TASKS
    return ORDER_INDEX
def original_task(position):
    return ORIGINAL_TASKS[int(position)]
def verify_mapping(cfg):
    install()
    from libero.libero.benchmark import get_benchmark
    a=get_benchmark(cfg.benchmark_name)(0)
    b=get_benchmark(cfg.benchmark_name)(ORDER_INDEX)
    assert all(a.get_task(i)==b.get_task(i) for i in range(6))
    assert all(b.get_task(i)==a.get_task(ORIGINAL_TASKS[i]) for i in range(10))
    return {str(i):dict(original_task=ORIGINAL_TASKS[i],name=b.get_task(i).name,language=b.get_task(i).language,bddl=b.get_task(i).bddl_file,initial_file=b.get_task(i).init_states_file) for i in range(10)}
