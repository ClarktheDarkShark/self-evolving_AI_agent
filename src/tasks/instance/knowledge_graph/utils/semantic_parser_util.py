# mypy: ignore-errors


class SemanticParserUtil:
    @staticmethod
    def _tokenize_lisp(lisp_string: str) -> list[str]:
        tokens: list[str] = []
        current: list[str] = []
        in_quote = False
        escape_next = False
        for ch in lisp_string:
            if in_quote:
                current.append(ch)
                if escape_next:
                    escape_next = False
                elif ch == "\\":
                    escape_next = True
                elif ch == '"':
                    in_quote = False
                continue
            if ch == '"':
                in_quote = True
                current.append(ch)
                continue
            if ch.isspace():
                if current:
                    tokens.append("".join(current))
                    current = []
                continue
            current.append(ch)
        if current:
            tokens.append("".join(current))
        return tokens

    @staticmethod
    def lisp_to_nested_expression(lisp_string: str) -> list:
        """
        Takes a logical form as a lisp string and returns a nested list representation of the lisp.
        For example, "(count (division first))" would get mapped to ['count', ['division', 'first']].
        """
        stack: list = []
        current_expression: list = []
        tokens = SemanticParserUtil._tokenize_lisp(lisp_string)
        for token in tokens:
            while token[0] == "(":
                nested_expression: list = []
                current_expression.append(nested_expression)
                stack.append(current_expression)
                current_expression = nested_expression
                token = token[1:]
            processed_token = token.replace(")", "")
            if (
                len(processed_token) >= 2
                and processed_token[0] == '"'
                and processed_token[-1] == '"'
            ):
                processed_token = bytes(
                    processed_token[1:-1], "utf-8"
                ).decode("unicode_escape")
            current_expression.append(processed_token)
            while token[-1] == ")":
                current_expression = stack.pop()
                token = token[:-1]
        return current_expression[0]

    @staticmethod
    def expression_to_lisp(expression) -> str:
        rtn = "("
        for i, e in enumerate(expression):
            if isinstance(e, list):
                rtn += SemanticParserUtil.expression_to_lisp(e)
            else:
                rtn += e
            if i != len(expression) - 1:
                rtn += " "
        rtn += ")"
        return rtn


def main():
    lisp = "(A ((B C) D E) F)"
    expression = SemanticParserUtil.lisp_to_nested_expression(lisp)
    print(expression)
    print(SemanticParserUtil.expression_to_lisp(expression))


if __name__ == "__main__":
    main()
