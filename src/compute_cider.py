#pip install git+https://github.com/salaniz/pycocoevalcap
#sudo apt update
#sudo apt install default-jre -y

from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
from collections import defaultdict

# 予測と正解（例として仮のデータを使ってます）
# あなたのコードでは generated_captions と references をここに使ってください
# 例:
# generated_captions = ["A diagram of a cell."]
# references = [["A diagram of the interior of a cell."]]

def compute_cider_score(generated_captions, references):
    """
    generated_captions: List[str]
    references: List[List[str]]
    """
    # COCO形式の辞書を作成
    gts = {}
    res = {}
    for i, (hyp, refs) in enumerate(zip(generated_captions, references)):
        gts[i] = [{"caption": ref} for ref in refs]  # 複数の参照キャプション
        res[i] = [{"caption": hyp}]                 # 1つの生成キャプション

    # Tokenize
    tokenizer = PTBTokenizer()
    gts_token = tokenizer.tokenize(gts)
    res_token = tokenizer.tokenize(res)

    # CIDErスコア計算
    scorer = Cider()
    score, scores = scorer.compute_score(gts_token, res_token)
    print(f"📊 CIDEr score: {score:.4f}")
    return score

# 使用例
# compute_cider_score(generated_captions, references)
