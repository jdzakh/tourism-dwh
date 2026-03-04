import os

import bcrypt
import streamlit as st
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

load_dotenv()


def get_engine():
    host = os.getenv("PG_HOST", "localhost")
    port = os.getenv("PG_PORT", "5432")
    db = os.getenv("PG_DB", "tourism_dwh")
    user = os.getenv("PG_USER", "tourism")
    pwd = os.getenv("PG_PASSWORD", "tourism")
    if host == "localhost":
        port = os.getenv("PG_PORT", "5433")
    url = f"postgresql+psycopg2://{user}:{pwd}@{host}:{port}/{db}"
    return create_engine(url, pool_pre_ping=True, connect_args={"connect_timeout": 10})


def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def check_password(password: str, password_hash: str) -> bool:
    return bcrypt.checkpw(password.encode("utf-8"), password_hash.encode("utf-8"))


def main():
    st.set_page_config(page_title="Личный кабинет", layout="wide")
    st.title("Личный кабинет")
    st.caption("Авторизация пользователей. Пароли хранятся в виде bcrypt-хеша.")

    engine = get_engine()
    if "user_profile" not in st.session_state:
        st.session_state.user_profile = None

    tab_login, tab_register = st.tabs(["Вход", "Регистрация"])

    with tab_login:
        login_email = st.text_input("Эл. почта", key="login_email")
        login_password = st.text_input("Пароль", type="password", key="login_password")
        if st.button("Войти", type="primary", key="login_btn"):
            with engine.begin() as conn:
                row = conn.execute(
                    text("SELECT email, role, password_hash FROM app.users WHERE email = :e"),
                    {"e": login_email.strip().lower()},
                ).fetchone()
            if not row:
                st.error("Пользователь не найден.")
            elif not check_password(login_password, row.password_hash):
                st.error("Неверный пароль.")
            else:
                st.session_state.user_profile = {"email": row.email, "role": row.role}
                st.success("Вход выполнен.")
                st.switch_page("app.py")

    with tab_register:
        reg_name = st.text_input("Имя и фамилия", key="reg_name")
        reg_email = st.text_input("Эл. почта", key="reg_email")
        reg_password = st.text_input("Пароль", type="password", key="reg_password")
        reg_role = st.radio(
            "Роль",
            ["Туроператор", "Отельер", "Авиаперевозчик", "Региональный офис"],
            horizontal=True,
            key="reg_role",
        )
        if st.button("Зарегистрироваться", type="primary", key="register_btn"):
            if not reg_email.strip() or "@" not in reg_email:
                st.error("Введите корректную эл. почту.")
            elif len(reg_password) < 6:
                st.error("Пароль должен быть не короче 6 символов.")
            else:
                try:
                    with engine.begin() as conn:
                        conn.execute(
                            text(
                                """
                                INSERT INTO app.users (name, email, role, password_hash)
                                VALUES (:n, :e, :r, :p)
                                """
                            ),
                            {"n": reg_name.strip(), "e": reg_email.strip().lower(), "r": reg_role, "p": hash_password(reg_password)},
                        )
                    st.session_state.user_profile = {"name": reg_name.strip(), "email": reg_email.strip().lower(), "role": reg_role}
                    st.success("Профиль создан.")
                    st.switch_page("app.py")
                except Exception:
                    st.error("Не удалось создать профиль. Возможно, эта почта уже используется.")

    if st.session_state.user_profile:
        profile = st.session_state.user_profile
        st.info(f"Вы вошли как: {profile.get('role')} | {profile.get('email', '')}".strip())
        if st.button("Выйти", key="logout_btn"):
            st.session_state.user_profile = None
            st.success("Вы вышли из аккаунта.")


if __name__ == "__main__":
    main()
