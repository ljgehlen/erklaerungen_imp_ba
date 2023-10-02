1. create and save models with create_imp_model()
2. extract and save gradients with save_grad_and_out_info()

Known problems:
	-pruning masks of loaded state dicts (get_model) show wrong compression rate but return correct weight distribution of iteration