_base_ = ["radcloud.py"]

model = dict(encoder_input_channels=1)

trainer = dict(
    model=model
)

