_base_ = ["radcloud.py"]

model = dict(encoder_input_channels=40)

trainer = dict(
    model=model
)

