class AgentError(Exception):
    pass


class AgentContextLimitError(AgentError):
    pass


class AgentOutOfMemoryError(AgentError):
    pass


class AgentUnknownError(AgentError):
    pass


class ModelError(Exception):
    pass


class ModelContextLimitError(ModelError):
    pass


class ModelOutOfMemoryError(ModelError):
    pass


class ModelUnknownError(ModelError):
    pass
