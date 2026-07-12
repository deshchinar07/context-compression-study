from peek import ContextMap, Operation


def main() -> None:
    context_map = ContextMap.initial()
    context_map = context_map.apply(
        [
            Operation(
                type="ADD",
                section="context_roadmap",
                content="This is a tiny local smoke test for PEEK.",
            )
        ]
    )

    context_map.save("tmp/test-map.peek.json")
    loaded = ContextMap.from_file("tmp/test-map.peek.json")

    print("PEEK imported and ran successfully.")
    print(f"Items found: {len(loaded.items())}")
    print(loaded.text)


if __name__ == "__main__":
    main()
