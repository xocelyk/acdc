from ioi_dataset import IOIDataset

def load_datasets(model, device):
    N = 50
    ioi_dataset = IOIDataset(
        prompt_type="mixed",
        N=N,
        tokenizer=model.tokenizer,
        prepend_bos=False,
        seed=1,
        device=str(device)
    )
    abc_dataset = ioi_dataset.gen_flipped_prompts("ABB->XYZ, BAB->XYZ")
    ioi_dataset.to(device)
    abc_dataset.to(device)
    return ioi_dataset, abc_dataset

