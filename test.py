def test_conversion():
    expected_converted = np.zeros((
        len(self.expected_output),
        len(self.results[0])
    ))
    print(f"Expected: {self.expected_output}")

    for i, output in enumerate(self.expected_output):
        print(f"i={i} ; output={output}")
        expected_converted[i][output] = 1

    self.expected_output = expected_converted

list = []