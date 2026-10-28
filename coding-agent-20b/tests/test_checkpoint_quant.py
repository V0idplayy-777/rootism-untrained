import pytest



from conftest import requires_torch







torch = pytest.importorskip("torch")











@requires_torch



def test_checkpoint_integrity_roundtrip(tmp_path, tiny_model):



    from training.checkpoint import load_verified, save_checkpoint_atomic, verify_checkpoint



    path = save_checkpoint_atomic({"step": 1, "model_state_dict": tiny_model.state_dict(),



                                   "optimizer_state_dict": {}}, str(tmp_path), 1)



    assert verify_checkpoint(str(path))["valid"]



    state = load_verified(str(path))



    assert state["step"] == 1



                                                    



    with open(path, "ab") as f:



        f.write(b"corrupt")



    assert not verify_checkpoint(str(path))["valid"]











@requires_torch



def test_quantization_and_distillation(tiny_model):



    from inference.quantize import load_weight_only_int8, weight_only_int8_state_dict



    from training.distillation import distillation_loss



    packed = weight_only_int8_state_dict(tiny_model)



    assert any(k.endswith(".scale") for k in packed)



    student = type(tiny_model)(tiny_model.config)



    load_weight_only_int8(student, packed)



    out = distillation_loss(torch.randn(2, 4, 320), torch.randn(2, 4, 320))



    assert torch.isfinite(out["loss"])



