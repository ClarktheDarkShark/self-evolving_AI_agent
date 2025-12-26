import docker
import mysql.connector
import random
import socket
import time
import json
import re
from docker.models import containers
from typing import Mapping, Optional

from src.self_evolving_agent.tool_registry import get_registry, ToolResult


_ACTION_PATTERN = re.compile(
    r"<action\s+name=\"(?P<name>[^\"]+)\">(?P<body>[\s\S]*?)</action>", re.MULTILINE
)


class DBBenchContainer:
    port = 13000
    password = "password"

    def __init__(self, image: str = "mysql:8.0"):
        self.deleted = False
        self.image = image
        self.client = docker.from_env()
        p = DBBenchContainer.port + random.randint(0, 10000)
        while self.is_port_open(p):
            p += random.randint(0, 20)
        self.port = p
        self.container: containers.Container = self.client.containers.run(
            image,
            name=f"mysql_{self.port}",
            environment={"MYSQL_ROOT_PASSWORD": self.password},
            ports={"3306": self.port},
            detach=True,
            tty=True,
            stdin_open=True,
            remove=True,
        )

        time.sleep(1)

        retry = 0
        while True:
            try:
                self.conn = mysql.connector.connect(
                    host="127.0.0.1",
                    user="root",
                    password=self.password,
                    port=self.port,
                    pool_reset_session=True,
                )
            except mysql.connector.errors.OperationalError:
                time.sleep(1)
            except mysql.connector.InterfaceError:
                if retry > 10:
                    raise
                time.sleep(5)
            else:
                break
            retry += 1

    def delete(self) -> None:
        self.container.stop()
        self.deleted = True

    def __del__(self) -> None:
        try:
            if not self.deleted:
                self.delete()
        except Exception:  # noqa
            pass

    def execute(
        self,
        multiple_sql: str,
        database: Optional[str] = None,
    ) -> str:
        if tool_result := self._maybe_execute_generated_tool(multiple_sql):
            return tool_result
        self.conn.reconnect()
        try:
            cursor = self.conn.cursor()
            if database:
                cursor.execute(f"use `{database}`;")
                cursor.fetchall()
            sql_list = multiple_sql.split(";")
            sql_list = [sql.strip() for sql in sql_list if sql.strip() != ""]
            result = ""
            for sql in sql_list:
                cursor.execute(sql)
                result = str(cursor.fetchall())
                self.conn.commit()
        except Exception as e:
            result = str(e)
        return result

    def _maybe_execute_generated_tool(self, action_payload: str) -> Optional[str]:
        """
        Check whether the action payload matches a generated tool.
        If so, execute it through the registry. Otherwise, return None.
        """
        match = _ACTION_PATTERN.search(action_payload)
        if match is None:
            return None
        tool_name = match.group("name")
        registry = get_registry()
        if not registry.has_tool(tool_name):
            return None
        args: list = []
        kwargs: Mapping[str, object] = {}
        body = match.group("body").strip()
        if body:
            try:
                payload = json.loads(body)
                if isinstance(payload, Mapping):
                    args = payload.get("args", [])
                    kwargs = payload.get("kwargs", {})
                elif isinstance(payload, list):
                    args = payload
                else:
                    args = [payload]
            except Exception:
                # Non-JSON payloads are passed as a single argument string.
                args = [body]
        invocation: ToolResult = registry.invoke_tool(tool_name, *args, **kwargs)
        if invocation.success:
            return str(invocation.output)
        return str(invocation.error)

    def is_port_open(
        self, port: int
    ) -> bool:  # noqa (The quality checker of the IDE is wrong)
        try:
            self.client.containers.get(f"mysql_{port}")
            return True
        except Exception:  # noqa
            pass

        # Create a socket object
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)  # use IPv4 and TCP
        try:
            # Try to connect to the specified port
            sock.connect(("localhost", port))
            # If the connection succeeds, the port is occupied
            return True
        except ConnectionRefusedError:
            # If the connection is refused, the port is not occupied
            return False
        finally:
            # Close the socket
            sock.close()
