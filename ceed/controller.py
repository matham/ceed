from threading import Thread
from twisted.internet import reactor, protocol


class ViewProtocol(protocol.Protocol):

    def dataReceived(self, data):
        self.transport.write(data)


class ViewThread(Thread):

    def run(self):
        factory = protocol.ServerFactory()
        factory.protocol = ViewProtocol
        reactor.listenTCP(8000, factory)
        reactor.run()
