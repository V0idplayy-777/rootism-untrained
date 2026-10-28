import sys



from pathlib import Path







sys.path.insert(0, str(Path(__file__).parent.parent))







import pytest







try:



    import torch              



    HAS_TORCH = True



except ImportError:



    HAS_TORCH = False







requires_torch = pytest.mark.skipif(not HAS_TORCH, reason="torch not installed")











@pytest.fixture()



def tiny_model():



    from model.config import get_tiny_config



    from model.model import CodingAgentModel



    torch.manual_seed(0)



    return CodingAgentModel(get_tiny_config())











@pytest.fixture()



def tokenizer():



    from model.tokenizer import CodeTokenizer



    return CodeTokenizer()



