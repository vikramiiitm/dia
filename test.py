try:
	import torch
except ModuleNotFoundError:
	print("PyTorch is not installed. Install it with: pip install torch")
	raise

print("Torch:", torch.__version__)
print("Built CUDA version:", torch.version.cuda)

is_cuda = torch.cuda.is_available()
print("CUDA available:", is_cuda)

if is_cuda:
	try:
		print("CUDA device count:", torch.cuda.device_count())
		cur = torch.cuda.current_device()
		print("Current device index:", cur)
		try:
			print("Current device name:", torch.cuda.get_device_name(cur))
		except Exception as e:
			print("Could not get device name:", e)
	except Exception as e:
		print("Error querying CUDA devices:", e)
else:
	print("No CUDA devices available or PyTorch built without CUDA.")

try:
	cudnn_ver = torch.backends.cudnn.version()
	cudnn_avail = torch.backends.cudnn.is_available()
except Exception:
	cudnn_ver = None
	cudnn_avail = False

print("cuDNN available:", cudnn_avail)
print("cuDNN version:", cudnn_ver)