from typing import List, Dict

class Solution:
    def tokenize_numbers(self, numbers: List[int], vocab: Dict[str, int]) -> List[List[str]]:
        # Tokenize each number using greedy left-to-right longest match.
        # Return a list of token lists showing how each number gets split.
        print('test')
        output= []
        for i in range(len(numbers)):
            num= numbers[i]
            num= str(num)
            tokens= self.greedy_tokenization(num, vocab)
            output.append(tokens)
        return output

    def count_tokens(self, text: str, vocab: Dict[str, int]) -> int:
        # Count how many tokens the text uses with greedy tokenization.
        # Use greedy left-to-right longest match.
        tokens= self.greedy_tokenization(text, vocab)
        return len(tokens)

    def fertility_score(self, text: str, vocab: Dict[str, int]) -> float:
        # Compute tokens-per-word ratio (fertility).
        # Higher = more expensive and less efficient.
        # Round to 4 decimal places.
        words= text.split(' ')
        tokens= self.greedy_tokenization(text, vocab)
        ratio= (len(tokens)*1.0)/(len(words)*1.0)
        return round(ratio,4)
    
    def greedy_tokenization(self, text: str, vocab: Dict[str, int])->List[str]:
        token= text
        tokens= []
        done= False
        while not done:
            if token in vocab:
                tokens.append(token)
                text= text.removeprefix(token)
                token= text
            else: token= token[:-1]
            if len(token)==0: done=True
        return tokens
