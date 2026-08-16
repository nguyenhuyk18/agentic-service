from aio_pika import Message
import aio_pika

class RabbitService:
    def __init__(self):
        self.connection = None
        self.channel = None

    async def connect(self):
        self.connection = (
            await aio_pika.connect_robust(
                "amqp://localhost:5672"
            )
        )

        self.channel = (
            await self.connection.channel()
        )

        print(
            "RabbitMQ connected"
        )
    