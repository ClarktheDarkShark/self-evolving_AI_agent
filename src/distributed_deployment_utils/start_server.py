import argparse
import re
import time
import multiprocessing as mp


class ServerStarterUtility:
    search_port_pattern = r"https?://[^:]+:(\d+)"
    search_prefix_pattern = r"https?://[^/]+(/.*)"

    @classmethod
    def extract_server_port(cls, server_address: str) -> int:
        port_match = re.match(cls.search_port_pattern, server_address)
        if port_match is None:
            raise ValueError(f"Port not found in server address: {server_address}")
        return int(port_match.group(1))

    @classmethod
    def extract_server_prefix(cls, server_address: str) -> str:
        prefix_match = re.match(cls.search_prefix_pattern, server_address)
        if prefix_match is None:
            raise ValueError(f"Prefix not found in server address: {server_address}")
        return prefix_match.group(1)


def _start_chat_history_item_factory_server(config_path: str) -> None:
    """
    Runs in its own process. Loads config and constructs the factory INSIDE the child process.
    """
    from src.factories import ChatHistoryItemFactoryServer
    from src.typings import GeneralInstanceFactory
    from src.utils import ConfigLoader, SingletonLogger
    from src.run_experiment import ConfigUtility, ConfigUtilityCaller

    raw_config = ConfigLoader().load_from(config_path)
    assignment_config, environment_config, logger_config, _ = (
        ConfigUtility.read_raw_config(raw_config, ConfigUtilityCaller.SERVER)
    )
    logger = SingletonLogger.get_instance(logger_config)

    assert environment_config.chat_history_item_factory_client is not None
    server_address = environment_config.chat_history_item_factory_client.parameters[
        "server_address"
    ]
    port = ServerStarterUtility.extract_server_port(server_address)
    prefix = ServerStarterUtility.extract_server_prefix(server_address)

    # The *instance* factory config is stored in the task definition
    task_instance_factory = assignment_config.task
    chat_hist_factory_instance_factory = GeneralInstanceFactory.model_validate(
        task_instance_factory.parameters["chat_history_item_factory"]
    )

    try:
        logger.info("Starting Chat History Item Factory Server (child process)...")
        factory = chat_hist_factory_instance_factory.create()
        ChatHistoryItemFactoryServer.start_server(factory, port, prefix)
    except Exception:
        logger.exception("Chat History Item Factory Server crashed")


def _start_task_server(config_path: str) -> None:
    """
    Runs in its own process. Loads config and constructs the task INSIDE the child process.
    """
    from src.tasks import TaskServer
    from src.utils import ConfigLoader, SingletonLogger
    from src.run_experiment import ConfigUtility, ConfigUtilityCaller

    raw_config = ConfigLoader().load_from(config_path)
    assignment_config, environment_config, logger_config, _ = (
        ConfigUtility.read_raw_config(raw_config, ConfigUtilityCaller.SERVER)
    )
    logger = SingletonLogger.get_instance(logger_config)

    assert environment_config.task_client is not None
    server_address = environment_config.task_client.parameters["server_address"]
    port = ServerStarterUtility.extract_server_port(server_address)
    prefix = ServerStarterUtility.extract_server_prefix(server_address)

    # Important: the task should use the *client* for chat_history_item_factory (not the local factory instance)
    assert environment_config.chat_history_item_factory_client is not None
    task_instance_factory = assignment_config.task
    task_instance_factory.parameters["chat_history_item_factory"] = (
        environment_config.chat_history_item_factory_client
    )

    try:
        logger.info("Starting Task Server (child process)...")
        task = task_instance_factory.create()
        TaskServer.start_server(task, port, prefix)
    except Exception:
        logger.exception("Task Server crashed")


def main() -> None:
    # Use spawn for safety now that we're not passing unpicklable objects
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, required=True)
    args = parser.parse_args()

    p1 = mp.Process(
        target=_start_chat_history_item_factory_server,
        args=(args.config_path,),
        daemon=True,
    )
    p2 = mp.Process(target=_start_task_server, args=(args.config_path,), daemon=True)

    p1.start()
    time.sleep(5)  # wait for chat history server
    p2.start()

    print("Both servers running. Ctrl+C to exit.")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        for p in (p2, p1):
            if p.is_alive():
                p.terminate()
                p.join()


if __name__ == "__main__":
    main()
