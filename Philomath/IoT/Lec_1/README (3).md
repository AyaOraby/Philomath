# 🛰️ Mosquitto MQTT Broker - Localhost Setup on Windows

This guide documents a working example of using the Mosquitto MQTT broker on **localhost (127.0.0.1)** using the command line in Windows.

---

## 🧰 Step 1: Free the Port (1883)

If Mosquitto fails to start due to port 1883 being used, use these commands to identify and stop the process:

```bash
netstat -aon | findstr :1883
```

Example output:

```
  TCP    127.0.0.1:1883     0.0.0.0:0         LISTENING       5212
  TCP    127.0.0.1:1883     127.0.0.1:49750   ESTABLISHED     5212
  TCP    127.0.0.1:49750    127.0.0.1:1883    ESTABLISHED     14172
  TCP    [::1]:1883         [::]:0            LISTENING       5212
```

Kill both processes:

```bash
taskkill /PID 14172 /F
taskkill /PID 5212 /F
```

Check again:

```bash
netstat -aon | findstr :1883
```

---

## ▶️ Step 2: Start the Broker

Navigate to the Mosquitto directory:

```bash
cd "C:\Program Files\mosquitto"
```

Start the broker with default config:

```bash
mosquitto -v
```

Example output:
```
mosquitto version 2.0.14 starting
Using default config.
Starting in local only mode. Connections will only be possible from clients running on this machine.
Opening ipv4 listen socket on port 1883.
```

---

## 📤 Step 3: Publish to a Topic

Choose a topic and message. For example:

- Topic: `ayaoraby`
- Message: `"hello"`

Run the publish command:

```bash
mosquitto_pub -h localhost -t <ayaoraby> -m "hello"
```
Or 
From MQTT 127.0.0.1
Publish the message on port 

---

## 📥 Step 4: Subscribe to a Topic

To listen to messages published to a topic, use:

```bash
mosquitto_sub -h localhost -t ayaoraby
```

Now, when someone publishes a message to `ayaoraby`, you will see:

```
hello
hello
hello
...
```

> ✅ If you accidentally use `<topic>` instead of a real topic name, you’ll get an error:
```bash
The syntax of the command is incorrect.
```

---

## ✅ Summary of Commands

| Action         | Command Example |
|----------------|------------------|
| Start Broker   | `mosquitto -v` |
| Subscribe      | `mosquitto_sub -h localhost -t ayaoraby` |
| Publish        | `mosquitto_pub -h localhost -t ayaoraby -m "hello"` |
| Check port     | `netstat -aon | findstr :1883` |
| Kill process   | `taskkill /PID <PID> /F` |

---

## 🔗 Official Docs

- [Mosquitto Docs](https://mosquitto.org/documentation/)
- [MQTT Protocol](https://mqtt.org/)
