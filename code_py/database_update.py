from database import get_connection
import thu_nghiem as th

  
conn = get_connection()
cur = conn.cursor()

# Câu SQL
sql = """
INSERT INTO attack_sessions (
        time, duration, count_botnet,
        tot_pkts, tot_bytes, src_bytes,
        dir_forward, dir_bidirectional, 
        proto_icmp, proto_tcp,  proto_udp,
        dTos_0_0 , dTos_10_0
    )
VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
"""

cur.execute(sql, (
   th.attack_time, float(th.dur), int(th.count_botnet),
   int(th.tot_pkts), int(th.tot_bytes), int(th.src_bytes),
   int(th.dir_forward), int(th.dir_bidirectional),
   int(th.proto_icmp), int(th.proto_tcp), int(th.proto_udp),
   float(th.dtos_0_0), float(th.dtos_10_0)
))

conn.commit()
cur.close()
conn.close()

print("Đã lưu dữ liệu vào database")
