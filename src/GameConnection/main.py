from server import BeemServer


def main():
    print("Starting connections")
    server = BeemServer()
    server.start()

if __name__ == "__main__":
    main()