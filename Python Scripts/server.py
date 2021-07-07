import asyncio
import base64
import time

# Bind the socket to the port
#server_address = ('localhost', 11000)
#server_address = ('192.168.1.2', 11002)
#erver_address = ('129.169.34.183', 11002)
#server_address = ('192.168.1.111', 11003)
#server_address = ('localhost', 11003)


directory = 'C:\\Users\\2020\\UNITY\\HololensComms\\Assets\\Images\\'

class Server:
    def __init__(self, ip_addr, port, buff_size):
        self.server_addr = ip_addr
        self.port = port

        self.buffer_size = buff_size
        self.image_data_buffer = []
        self.image_data_bytes_recvd = 0


    async def tcp_echo_client(self, message, loop):
        reader, writer = await asyncio.open_connection(self.server_addr, self.port, loop=loop)

        print('Send: %r' % message)
        writer.write(message.encode())

        while True:
            try:
                data = await asyncio.wait_for(reader.read(self.buffer_size), timeout=5)
                #print('Received: %r' % data.decode())
                self.image_data_buffer.append(data)
                self.image_data_bytes_recvd += len(data)
                print('Bytes received: %d' % len(data))
            except asyncio.TimeoutError:
                print('Timeout!')
                break

        print('Closing socket')
        writer.close()
        

    def write_image_file(self, directory):
        img_meta_data = ""
        timestamp = ""
        
        print('Image data buffer: %d' % len(self.image_data_buffer))
        print('Image data bytes received: %d' % self.image_data_bytes_recvd)
        self.image_data_concat = b''.join(self.image_data_buffer)

        t = time.time()
        timestamp = str(round(t))

        img_filename = directory + '\\context_img_%s.png' % timestamp
        print('Saving img to %s' % img_filename)
        with open(img_filename, "wb") as fh:
            fh.write(base64.decodebytes(self.image_data_concat))
            fh.close()
        
        log_filename = directory + '\\context_img_%s.log' % timestamp                      
        print('Saving img meta data to %s' % log_filename)
        with open(log_filename, "wb") as fh:
            fh.write(img_meta_data.encode())
            fh.close()
        
        buffer_filename = directory + '\\context_img_buff_%s.log' % timestamp  
        with open(buffer_filename, "wb") as fh:
            for b in self.image_data_buffer:
                fh.write(b)
            fh.close()


def main():
    server = Server('169.254.155.249', 9090, 10800)
    message = 'Starting up server'
    loop = asyncio.get_event_loop()
    server.tcp_echo_client(message, loop)
    loop.run_until_complete(server.tcp_echo_client(message, loop))
    loop.close()
    server.write_image_file(directory)

    #img_dim = [504, 896]
    #patch_dim = [63, 112]
    #label_dim = [0.2,0.05]
    #adapt = ad.Adapt(np.array(img_dim), np.array(patch_dim), np.array(label_dim))

    #(labelPos, uvPlace) = adapt.place(server.image_data_concat)
    #(labelColor, textColor) = adapt.color(uvPlace)

    #adaptPayload = adapt.msgString(labelPos, labelColor, textColor)
    #print(adaptPayload)
    
    #out_filename = directory + 'out.txt'
    #print('Saving info to %s' % out_filename)
    #with open(out_filename, "w") as f:
    #    line = str(label_dim[0]) + ',' + str(label_dim[1]) + ';' + adaptPayload
    #    f.write(line)
    
    #f.close()
    

if __name__ == "__main__":
    main()
