from remi_z.legacy_tokenizer import RemiTokenizer

def test_old_tokenizer():
    pass

import unittest

class TestCoreFunctions(unittest.TestCase):
    
    def test_add(self):
        tk = RemiTokenizer()

        # Testing add function
        self.assertEqual(add(2, 3), 5)  # Expected result: 2 + 3 = 5
        self.assertEqual(add(-1, 1), 0) # Expected result: -1 + 1 = 0
        self.assertEqual(add(0, 0), 0)  # Expected result: 0 + 0 = 0

if __name__ == '__main__':
    unittest.main()
