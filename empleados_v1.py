import sqlite3

class Empleados:

    def abrir(self,base_de_datos):
        conexion=sqlite3.connect(base_de_datos)
        return conexion


    def alta(self, base_de_datos,datos,tipo):
        cone=self.abrir(base_de_datos)
        cursor=cone.cursor()
        if tipo=="T":
            sql="insert into empleados(DNI,NOMBRE, APELLIDOS, CIF, TELEFONO, MAIL, departamento, cargo ) values (?,?,?,?,?,?,?,?)"
        else:
            sql="insert into visitas(DNI,NOMBRE, APELLIDOS, CIF, TELEFONO, MAIL, departamento, cargo ) values (?,?,?,?,?,?,?,?)"
        cursor.execute(sql, datos)
        cone.commit()
        cone.close()

    def consulta(self, base_de_datos,datos):
        try:
            cone=self.abrir(base_de_datos)
            cursor=cone.cursor()
            sql="select * from empleados where DNI=?"
            cursor.execute(sql, datos)
            return cursor.fetchall()
        finally:
            cone.close()
            
    def recuperar_todos(self, base_de_datos):
        try:
            cone=self.abrir(base_de_datos)
            cursor=cone.cursor()
            sql="select * from empleados"
            cursor.execute(sql)
            return cursor.fetchall()
        finally:
            cone.close()