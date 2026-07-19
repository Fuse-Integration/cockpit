import { defaultJoystickCalibration } from '@/assets/defaults'
import {
  type ElectronSDLJoystickControllerStateEventData as ElectronSDLControllerStateEventData,
  type JoystickCalibration,
  type JoystickState,
  convertSDLControllerStateToGamepadState,
  convertSDLJoystickStateToGamepadState,
} from '@/types/joystick'
import { JoystickMapVidPid, JoystickModel } from '@/types/joystick-model-defs'

import { settingsManager } from '../settings-management'
import { isElectron } from '../utils'
import { applyCalibration } from './calibration'

export { JoystickModel }

export const joystickCalibrationOptionsKey = 'cockpit-joystick-calibration-options'

// Reserved index for the synthetic keyboard joystick. Physical gamepads use small indices (0-3),
// so a high constant avoids any collision and lets us exempt it from the Gamepad-API cull.
const keyboardJoystickIndex = 1000
const keyboardJoystickId = 'Cockpit Virtual Keyboard (STANDARD GAMEPAD)'
// Ramp rates (units/second) so held keys accelerate smoothly and release decays quickly, giving
// digital keys an analog-like feel. Emit rate is high enough for responsive control.
const keyboardRampUpPerSec = 3
const keyboardEmitIntervalMs = 50
// A held key fires keydown repeatedly (auto-repeat). If those repeats stop but no keyup arrived
// (keyup can be missed/intercepted, leaving the axis stuck and the vehicle driving), auto-release
// the key this long after its last keydown. Only applied once a key has actually repeated, so the
// initial pre-repeat delay of a fresh press does not trigger a false release.
const keyboardStuckKeyMs = 150
// Backstop for keys that never repeated (a quick tap whose keyup was missed): release them once
// they exceed the longest plausible OS auto-repeat delay, by which point a genuinely-held key
// would already be repeating and covered by the faster threshold above.
const keyboardStuckKeyInitialMs = 700
// Map of key codes to [axisIndex, direction] on the standard gamepad layout.
// axis1 drives forward/back (mapped to MAVLink axis_x, +1 = forward), axis2 drives yaw
// (mapped to axis_r, +1 = right). Up/W = +1 so forward matches the vehicle's forward.
const keyboardAxisBindings: Record<string, [number, number]> = {
  ArrowUp: [1, 1],
  KeyW: [1, 1],
  ArrowDown: [1, -1],
  KeyS: [1, -1],
  ArrowLeft: [2, -1],
  KeyA: [2, -1],
  ArrowRight: [2, 1],
  KeyD: [2, 1],
}

/**
 * Possible events from GamepadListener
 * https://developer.mozilla.org/en-US/docs/Web/API/Gamepad_API/Using_the_Gamepad_API
 */
export enum EventType {
  Unknown = 'unknown',
  Connected = 'connected',
  Disconnected = 'disconnected',
  Axis = 'axis',
  Button = 'button',
}

// Necessary to add functions
// eslint-disable-next-line @typescript-eslint/no-namespace
export namespace EventType {
  /**
   * Return a list of possible events for gamepad api
   * @returns {Array<string>}
   */
  export function events(): Array<string> {
    return Object.keys(EventType).map((name) => name.toLowerCase())
  }

  /**
   * Return enum value from Gamepad API event
   * @param {string} type
   * @returns {EventType}
   */
  export function fromGamepadEventType(type: string): EventType {
    const fields = type.split(':')
    if (fields.length <= 1) {
      return EventType.Unknown
    }

    const name = fields[1]

    for (const eventName of EventType.events()) {
      if (eventName == name) {
        return eventName as EventType
      }
    }

    return EventType.Unknown
  }
}

// Encapsulate joystick event values
// eslint-disable-next-line @typescript-eslint/no-namespace
export namespace JoystickDetail {
  /**
   * Stick values
   */
  export enum Stick {
    Left = 0,
    Right = 1,
  }

  /**
   * Axis values
   */
  export enum Axis {
    Horizontal = 0,
    Vertical = 1,
  }
}

export type JoystickAxisEvent = {
  /**
   * Event type
   */
  type: EventType.Axis
  /**
   * Detail information about the event
   */
  detail: {
    /**
     * Joystick index
     */
    index: number
    /**
     * Gamepad object
     */
    gamepad: Gamepad
    /**
     * Stick position
     */
    stick: JoystickDetail.Stick
    /**
     * Axis type
     */
    axis: JoystickDetail.Axis
    /**
     * Axis value
     */
    value: number
  }
}

export type JoystickStateEvent = {
  /**
   * Joystick index
   */
  index: number
  /**
   * Gamepad object holding the raw, un-calibrated axes and buttons.
   * Consumers that need the actual stick position (e.g. the calibration UI
   * or the joystick visualizations) should read from here.
   */
  gamepad: Gamepad
  /**
   * Joystick state with calibration (deadband, exponential) already applied.
   * Consumers driving vehicle commands should read from here.
   */
  calibratedState: JoystickState
}

export type JoystickConnectEvent = {
  /**
   * Event type
   */
  type: EventType.Connected
  /**
   * Detail information about the event
   */
  detail: {
    /**
     * Joystick index
     */
    index: number
    /**
     * Gamepad object
     */
    gamepad: Gamepad
  }
}

export type JoystickDisconnectEvent = {
  /**
   * Event type
   */
  type: EventType.Disconnected
  /**
   * Detail information about the event
   */
  detail: {
    /**
     * Joystick index
     */
    index: number
  }
}

export type GamepadState = {
  /**
   * The epoch of the given state
   */
  timestamp: number
  /**
   * The actual state
   */
  state: {
    /**
     * Gamepad axes state
     */
    axes: number[]
    /**
     * Gamepad buttons state
     */
    buttons: GamepadButton[]
  }
}

export type JoysticksMap = Map<number, Gamepad>

export type JoystickConnectionEvent = JoystickConnectEvent | JoystickDisconnectEvent

type CallbackJoystickStateEventType = (event: JoystickStateEvent) => void
type CallbackJoystickConnectionEventType = (event: JoysticksMap) => void

/**
 * Joystick Manager
 * Abstraction over Gamepad API and SDL
 */
class JoystickManager {
  private static instance = new JoystickManager()

  private callbacksJoystickConnection: Array<CallbackJoystickConnectionEventType> = []
  private callbacksJoystickState: Array<CallbackJoystickStateEventType> = []
  private joysticks: JoysticksMap = new Map()
  private enabledJoysticks: Array<number> = []
  private animationFrameId: number | null = null
  private lastTimeGamepadStatesPolled = 0
  private previousGamepadState: Map<number, JoystickState> = new Map()
  private calibrationOptions: Map<JoystickModel, JoystickCalibration> = new Map()
  private keyboardEnabled = false
  private keyboardPressedKeys: Set<string> = new Set()
  private keyboardLastKeydown: Map<string, number> = new Map()
  private keyboardRepeating: Set<string> = new Set()
  private keyboardAxes = [0, 0, 0, 0]
  private keyboardEmitTimer: ReturnType<typeof setInterval> | null = null
  private keyboardLastTick = 0
  private keyboardKeyDownHandler: ((e: KeyboardEvent) => void) | null = null
  private keyboardKeyUpHandler: ((e: KeyboardEvent) => void) | null = null
  private keyboardBlurHandler: (() => void) | null = null
  /**
   * Singleton constructor
   */
  private constructor() {
    console.log('Starting JoystickManager...')

    if (isElectron()) {
      // Also check SDL status directly after a short delay
      setTimeout(async () => {
        try {
          if (!window.electronAPI) {
            console.error('Electron API not available.')
            this.startGamepadApiMonitoringRoutine()
            return
          }
          const status = await window.electronAPI.checkSDLStatus()

          if (!status.loaded) {
            console.error('SDL not loaded according to status check, falling back to Gamepad API.')
            this.startGamepadApiMonitoringRoutine()
          } else {
            console.log('SDL loaded successfully, using SDL for controllers.')
            this.startElectronSdlJoystickMonitoringRoutine()
          }
        } catch (error) {
          console.error('Error checking SDL status, falling back to Gamepad API:', error)
          this.startGamepadApiMonitoringRoutine()
        }
      }, 3000)
    } else {
      // In browser or if electronAPI is not available, use the Gamepad API
      console.log('Not in Electron or Electron API not available, using Gamepad API.')
      this.startGamepadApiMonitoringRoutine()
    }
  }

  /**
   * Singleton access
   * @returns {JoystickManager}
   */
  static self(): JoystickManager {
    return JoystickManager.instance
  }

  /**
   * Callback to be used and receive joystick connection updates
   * @param {JoysticksMap} callback
   */
  onJoystickConnectionUpdate(callback: CallbackJoystickConnectionEventType): void {
    this.callbacksJoystickConnection.push(callback)
    // Seed the new subscriber with the current state. Connection events only fire on change, so a
    // component that mounts after a device connected (e.g. the keyboard virtual joystick, or a
    // gamepad connected earlier) would otherwise never learn about it and stay in a disconnected UI.
    if (this.joysticks.size > 0) {
      callback(this.joysticks)
    }
  }

  /**
   * Get Vendor ID and Product ID from joystick
   * @param {string} gamepadId Id of the gamepad
   * @returns {'vendor_id: string | undefined, product_id: string | undefined'} VID and PID
   */
  getVidPid(gamepadId: string): {
    vendor_id: string | undefined // eslint-disable-line
    product_id: string | undefined // eslint-disable-line
  } {
    const vendor_regex = new RegExp('Vendor: (?<vendor_id>[0-9a-f]{4})')
    const product_regex = new RegExp('Product: (?<product_id>[0-9a-f]{4})')
    const vendor_id = vendor_regex.exec(gamepadId)?.groups?.vendor_id
    const product_id = product_regex.exec(gamepadId)?.groups?.product_id
    return { vendor_id, product_id }
  }

  /**
   * Get joystick model
   * @param {string} gamepadId Id of the gamepad
   * @returns {JoystickModel} Joystick model
   */
  getModel(gamepadId: string): JoystickModel {
    if (gamepadId === keyboardJoystickId) {
      return JoystickModel.VirtualKeyboard
    }

    const { vendor_id, product_id } = this.getVidPid(gamepadId)

    if (vendor_id == undefined || product_id == undefined) {
      return JoystickModel.Unknown
    }
    return JoystickMapVidPid.get(`${vendor_id}:${product_id}`) ?? JoystickModel.Unknown
  }

  /**
   * Register joystick event callback
   * @param {callbackJoystickStateEventType} callback
   */
  onJoystickStateUpdate(callback: CallbackJoystickStateEventType): void {
    this.callbacksJoystickState.push(callback)
  }

  /**
   * Check SDL status
   * @returns {Promise<void>}
   */
  private async startSDLStatusCheckRoutine(): Promise<void> {
    if (!window.electronAPI) {
      return
    }
    const status = await window.electronAPI.checkSDLStatus()

    if (!status.loaded) {
      console.error('SDL connection dropped. Falling back to Gamepad API.')
      this.startGamepadApiMonitoringRoutine()
    }

    // Remove any joysticks that are not in the status.connectedControllers map
    let joystickConnectionsChanged = false
    for (const [, gamepad] of this.joysticks) {
      if (!status.connectedControllers.has(gamepad.index) && !status.connectedJoysticks.has(gamepad.index)) {
        this.joysticks.delete(gamepad.index)
        joystickConnectionsChanged = true
      }
    }

    if (joystickConnectionsChanged) {
      this.emitJoystickConnectionUpdate()
    }

    setTimeout(() => this.startSDLStatusCheckRoutine(), 1000)
  }

  /**
   * Set up joystick monitoring in Electron environment
   * This method sets up listeners for joystick events from the main process
   * and converts them to the same format as the Gamepad API events
   */
  private startElectronSdlJoystickMonitoringRoutine(): void {
    if (!window.electronAPI) {
      console.error('Electron API not available.')
      return
    }

    // Check calibration settings every second
    this.updateCalibrationSettings()

    // Start checking SDL status
    this.startSDLStatusCheckRoutine()

    /**
     * Listen for joystick state updates from the main process
     * Converts SDL joystick state to Gamepad API format
     * @param data The joystick state data from the main process
     */
    window.electronAPI.onElectronSDLControllerJoystickStateChange((data: ElectronSDLControllerStateEventData) => {
      // Convert SDL joystick state to our event format

      const gamepadState =
        data.type === 'joystick'
          ? convertSDLJoystickStateToGamepadState(data.state)
          : convertSDLControllerStateToGamepadState(data.state)

      const gamepadId = `${data.deviceName} (SDL STANDARD JOYSTICK Vendor: ${data.vendorId} Product: ${data.productId})`
      const gamepadModel = this.getModel(gamepadId)

      const rawAxes = gamepadState.axes.map((value) => value ?? 0)
      const rawButtons = gamepadState.buttons.map((value) => value ?? 0)

      const joystickEvent: JoystickStateEvent = {
        index: data.deviceId,
        gamepad: {
          id: gamepadId,
          index: data.deviceId,
          connected: true,
          timestamp: Date.now(),
          mapping: 'standard',
          axes: rawAxes,
          buttons: rawButtons.map((value) => ({
            pressed: value > 0.5,
            value: value,
            touched: false,
          })),
          vibrationActuator: {
            playEffect: async () => 'complete' as const,
            reset: async () => 'complete' as const,
          },
        },
        calibratedState: this.buildCalibratedState(rawAxes, rawButtons, gamepadModel),
      }

      // Add joystick to the list of joysticks if it is not already there
      if (!this.joysticks.has(data.deviceId)) {
        this.joysticks.set(data.deviceId, joystickEvent.gamepad)
        this.enabledJoysticks.push(data.deviceId)
        // Emit joystick connection update only when a new joystick is detected
        this.emitJoystickConnectionUpdate()
      }

      // Emit joystick state update
      this.emitStateEvent(joystickEvent)
    })
  }

  /**
   * Poll for gamepad connections and disconnection every 500ms, and activates polling the gamepad states.
   * The polling for connections and disconnections is a workaround to get around the fact that the gamepad API events do not work the same way in all browsers.
   * In Chrome, for example, the gamepadconnected event is sometimes not fired when a gamepad is connected after a long time since the page was loaded.
   * This is a workaround to get around this issue.
   */
  private startGamepadApiMonitoringRoutine(): void {
    // Start polling for gamepad connections and disconnections
    this.pollGamepadsConnections(500)

    // Start polling for gamepad states
    this.pollGamepadsStates(20)

    // Check calibration settings every second
    this.updateCalibrationSettings()
  }

  /**
   * Update calibration settings
   */
  private updateCalibrationSettings(): void {
    this.loadCalibrationSettings()
    this.syncKeyboardJoystickFromSettings()
    setTimeout(() => {
      this.updateCalibrationSettings()
    }, 1000)
  }

  /**
   * Enable/disable the keyboard joystick to match the persisted opt-in setting. Runs on the same
   * periodic loop as calibration so the device activates on app boot (not only when the joystick
   * configuration page is opened) and reflects the setting even without that view mounted.
   */
  private syncKeyboardJoystickFromSettings(): void {
    const enabled = settingsManager.getKeyValue('cockpit-keyboard-joystick-enabled') === true
    if (enabled !== this.keyboardEnabled) {
      this.setKeyboardJoystickEnabled(enabled)
    }
  }

  /**
   * Poll for gamepad connections and disconnections every interval ms
   * @param {number} interval The interval in milliseconds
   */
  private pollGamepadsConnections(interval: number): void {
    const gamepadConnectionsState = navigator.getGamepads()

    let joystickConnectionsChanged = false

    // Add new gamepads to the list
    for (const gamepad of gamepadConnectionsState) {
      if (gamepad && !this.joysticks.has(gamepad.index)) {
        this.joysticks.set(gamepad.index, gamepad)
        this.enabledJoysticks.push(gamepad.index)
        console.log(`Joystick ${gamepad.index} connected.`)
        joystickConnectionsChanged = true

        // Log some information about the joystick so we can track used joysticks and easily add more to our database
        try {
          console.log(`Joystick info:
            name: '${gamepad.id}'
            id: '${gamepad.index}'
            vendor: '${this.getVidPid(gamepad.id).vendor_id}'
            product: '${this.getVidPid(gamepad.id).product_id}'
            axes: ${gamepad.axes.join(', ')}
            buttons: ${gamepad.buttons.map((button) => button.value).join(', ')}
          `)
        } catch (error) {
          console.error(`Error logging joystick info for '${gamepad.id}' with index '${gamepad.index}':`, error)
        }
      }
    }

    // Remove gamepads that are not connected anymore. The synthetic keyboard joystick is not part of
    // navigator.getGamepads(), so it must be exempted or it would be culled on every poll.
    for (const gamepad of this.joysticks.values()) {
      if (gamepad.index === keyboardJoystickIndex) continue
      if (!gamepadConnectionsState.map((g) => g?.index).includes(gamepad.index)) {
        this.joysticks.delete(gamepad.index)
        this.enabledJoysticks = this.enabledJoysticks.filter((index) => index !== gamepad.index)
        joystickConnectionsChanged = true
      }
    }

    // Emit the updated list of joysticks for all listeners if there were any changes
    if (joystickConnectionsChanged) {
      this.emitJoystickConnectionUpdate()
    }

    setTimeout(() => {
      this.pollGamepadsConnections(interval)
    }, interval)
  }

  /**
   * Load calibration settings from local storage
   */
  private loadCalibrationSettings(): void {
    try {
      const stored = settingsManager.getKeyValue(joystickCalibrationOptionsKey)
      if (stored !== undefined) {
        const options = stored as Record<JoystickModel, JoystickCalibration>
        this.calibrationOptions = new Map(Object.entries(options).map(([key, value]) => [key as JoystickModel, value]))
      }
    } catch (error) {
      console.error('Failed to load joystick calibration settings:', error)
    }
  }

  /**
   * Build a calibrated `JoystickState` from raw axes and buttons.
   * Calibration is per-axis: deadband and exponential scaling are applied
   * using the thresholds/factors stored for the given joystick model.
   * Buttons are passed through untouched, matching the previous behavior.
   * @param {number[]} rawAxes Raw axis values straight from the device
   * @param {number[]} rawButtons Raw button values straight from the device
   * @param {JoystickModel} model Joystick model used to look up calibration
   * @returns {JoystickState} The calibrated joystick state
   */
  private buildCalibratedState(rawAxes: number[], rawButtons: number[], model: JoystickModel): JoystickState {
    const calibration = this.calibrationOptions.get(model) ?? defaultJoystickCalibration
    return {
      axes: rawAxes.map((value, index) => applyCalibration('axis', index, value, calibration)),
      buttons: [...rawButtons],
    }
  }

  /**
   * Poll for gamepad state changes
   * @param {number} interval The interval in milliseconds
   */
  private pollGamepadsStates(interval: number): void {
    if (new Date().getTime() - this.lastTimeGamepadStatesPolled < interval) return

    const gamepads = navigator.getGamepads()

    for (const gamepad of gamepads) {
      if (!gamepad) continue

      const joystickModel = this.getModel(gamepad.id)

      const previousState = this.previousGamepadState.get(gamepad.index)

      const rawAxes = [...gamepad.axes]
      const rawButtons = gamepad.buttons.map((button) => button.value)
      let shouldEmitStateEvent = false

      if (previousState) {
        rawAxes.forEach((value, index) => {
          if (previousState.axes[index] !== value) {
            shouldEmitStateEvent = true
          }
        })

        rawButtons.forEach((value, index) => {
          if (previousState.buttons[index] !== value) {
            shouldEmitStateEvent = true
          }
        })
      }

      // Update previous state
      this.previousGamepadState.set(gamepad.index, { axes: rawAxes, buttons: rawButtons })

      if (shouldEmitStateEvent) {
        this.emitStateEvent({
          index: gamepad.index,
          gamepad: gamepad,
          calibratedState: this.buildCalibratedState(rawAxes, rawButtons, joystickModel),
        })
      }
    }

    // Continue polling
    this.animationFrameId = requestAnimationFrame(() => this.pollGamepadsStates(interval))
  }

  /**
   * Emit state event to registered callbacks
   * @param {JoystickStateEvent} joystickEvent - The state event to emit
   */
  private emitStateEvent(joystickEvent: JoystickStateEvent): void {
    if (!this.enabledJoysticks.includes(joystickEvent.index)) return

    // Get the joystick model to check if it's disabled
    const model = this.getModel(joystickEvent.gamepad.id)
    const disabledJoystickModels = settingsManager.getKeyValue('cockpit-disabled-joystick-models') ?? []
    if (disabledJoystickModels.includes(model)) return

    // Emit state event to registered callbacks
    for (const callback of this.callbacksJoystickState) {
      callback(joystickEvent)
    }
  }

  /**
   * Emit joystick state updates to registered callbacks
   * @param {JoystickStateEvent} joystickEvent - The state event to emit
   */
  /**
   * Emit joystick connection updates to registered callbacks
   */
  private emitJoystickConnectionUpdate(): void {
    for (const callback of this.callbacksJoystickConnection) {
      callback(this.joysticks)
    }
  }

  /**
   * Whether the virtual keyboard joystick is currently enabled
   * @returns {boolean}
   */
  isKeyboardJoystickEnabled(): boolean {
    return this.keyboardEnabled
  }

  /**
   * Enable or disable the virtual keyboard joystick. When enabled, arrow keys / WASD drive the
   * standard-gamepad axes (left-stick Y for forward/back, right-stick X for yaw), letting operators
   * fly manual control without a physical gamepad. The device appears as a normal standard joystick,
   * so the existing mapping/calibration/forwarding pipeline treats it like any other controller.
   * @param {boolean} enabled Whether the keyboard joystick should be active
   */
  setKeyboardJoystickEnabled(enabled: boolean): void {
    if (enabled === this.keyboardEnabled) return
    this.keyboardEnabled = enabled
    enabled ? this.attachKeyboardJoystick() : this.detachKeyboardJoystick()
  }

  /**
   * Build the synthetic Gamepad object representing the current keyboard axis state
   * @returns {Gamepad} A standard-mapping gamepad snapshot
   */
  private buildKeyboardGamepad(): Gamepad {
    return {
      id: keyboardJoystickId,
      index: keyboardJoystickIndex,
      connected: true,
      timestamp: performance.now(),
      mapping: 'standard',
      axes: [...this.keyboardAxes],
      buttons: Array.from({ length: 18 }, () => ({ pressed: false, value: 0, touched: false })),
      vibrationActuator: {
        playEffect: async () => 'complete' as const,
        reset: async () => 'complete' as const,
      },
    } as unknown as Gamepad
  }

  /**
   * Register the keyboard device, attach key listeners, and start the ramp/emit loop
   */
  private attachKeyboardJoystick(): void {
    this.clearKeyboardKeys()
    this.keyboardAxes = [0, 0, 0, 0]

    this.keyboardKeyDownHandler = (e: KeyboardEvent) => {
      const isStopKey = e.code === 'Space'
      if (!isStopKey && !(e.code in keyboardAxisBindings)) return
      // Ignore when typing in an input/textarea so keyboard driving never hijacks form entry.
      const target = e.target as HTMLElement | null
      if (target && (target.isContentEditable || ['INPUT', 'TEXTAREA', 'SELECT'].includes(target.tagName))) return
      e.preventDefault()
      if (isStopKey) {
        // Space is a hard stop: drop all held movement keys and snap axes to zero immediately,
        // bypassing the ramp so the vehicle halts at once.
        this.clearKeyboardKeys()
        this.keyboardAxes = [0, 0, 0, 0]
        this.emitStateEvent({
          index: keyboardJoystickIndex,
          gamepad: this.buildKeyboardGamepad(),
          calibratedState: this.buildCalibratedState([0, 0, 0, 0], [], JoystickModel.VirtualKeyboard),
        })
        return
      }
      // A keydown for an already-pressed key is the OS auto-repeat — mark it so the stuck-key
      // watchdog can safely release it once repeats stop.
      if (this.keyboardPressedKeys.has(e.code)) this.keyboardRepeating.add(e.code)
      this.keyboardPressedKeys.add(e.code)
      this.keyboardLastKeydown.set(e.code, performance.now())
    }
    this.keyboardKeyUpHandler = (e: KeyboardEvent) => {
      if (this.keyboardPressedKeys.delete(e.code)) e.preventDefault()
      this.keyboardRepeating.delete(e.code)
      this.keyboardLastKeydown.delete(e.code)
    }
    // Releasing focus (alt-tab, clicking away) must stop the vehicle, matching gamepad safety.
    this.keyboardBlurHandler = () => this.clearKeyboardKeys()

    window.addEventListener('keydown', this.keyboardKeyDownHandler)
    window.addEventListener('keyup', this.keyboardKeyUpHandler)
    window.addEventListener('blur', this.keyboardBlurHandler)

    this.joysticks.set(keyboardJoystickIndex, this.buildKeyboardGamepad())
    if (!this.enabledJoysticks.includes(keyboardJoystickIndex)) {
      this.enabledJoysticks.push(keyboardJoystickIndex)
    }
    this.emitJoystickConnectionUpdate()

    this.keyboardLastTick = performance.now()
    this.keyboardEmitTimer = setInterval(() => this.tickKeyboardJoystick(), keyboardEmitIntervalMs)
  }

  /**
   * Stop the keyboard device: remove listeners, zero the axes, and unregister it
   */
  private detachKeyboardJoystick(): void {
    if (this.keyboardKeyDownHandler) window.removeEventListener('keydown', this.keyboardKeyDownHandler)
    if (this.keyboardKeyUpHandler) window.removeEventListener('keyup', this.keyboardKeyUpHandler)
    if (this.keyboardBlurHandler) window.removeEventListener('blur', this.keyboardBlurHandler)
    this.keyboardKeyDownHandler = null
    this.keyboardKeyUpHandler = null
    this.keyboardBlurHandler = null

    if (this.keyboardEmitTimer !== null) {
      clearInterval(this.keyboardEmitTimer)
      this.keyboardEmitTimer = null
    }

    this.clearKeyboardKeys()
    this.keyboardAxes = [0, 0, 0, 0]
    // Emit one final zeroed state so consumers stop the vehicle before the device disappears.
    this.emitStateEvent({
      index: keyboardJoystickIndex,
      gamepad: this.buildKeyboardGamepad(),
      calibratedState: this.buildCalibratedState([0, 0, 0, 0], [], JoystickModel.VirtualKeyboard),
    })

    this.joysticks.delete(keyboardJoystickIndex)
    this.enabledJoysticks = this.enabledJoysticks.filter((index) => index !== keyboardJoystickIndex)
    this.emitJoystickConnectionUpdate()
  }

  /**
   * Clear all keyboard key-tracking state (pressed, repeat, and last-keydown timers)
   */
  private clearKeyboardKeys(): void {
    this.keyboardPressedKeys.clear()
    this.keyboardRepeating.clear()
    this.keyboardLastKeydown.clear()
  }

  /**
   * Release keys that are stuck: a key that had been auto-repeating but whose repeats have stopped
   * without a keyup (missed/intercepted keyup) would otherwise hold the axis and keep the vehicle
   * moving. Only keys that have actually repeated are eligible, so a fresh press awaiting its first
   * repeat is never falsely released.
   * @param {number} now Current timestamp (performance.now)
   */
  private releaseStuckKeyboardKeys(now: number): void {
    for (const code of [...this.keyboardPressedKeys]) {
      const last = this.keyboardLastKeydown.get(code) ?? 0
      const threshold = this.keyboardRepeating.has(code) ? keyboardStuckKeyMs : keyboardStuckKeyInitialMs
      if (now - last > threshold) {
        this.keyboardPressedKeys.delete(code)
        this.keyboardRepeating.delete(code)
        this.keyboardLastKeydown.delete(code)
      }
    }
  }

  /**
   * One ramp/emit tick: move each axis toward its target (held keys) and emit the state if changed
   */
  private tickKeyboardJoystick(): void {
    const now = performance.now()
    const dt = Math.min((now - this.keyboardLastTick) / 1000, 0.25)
    this.keyboardLastTick = now

    this.releaseStuckKeyboardKeys(now)

    const targets = [0, 0, 0, 0]
    for (const code of this.keyboardPressedKeys) {
      const [axis, dir] = keyboardAxisBindings[code]
      targets[axis] += dir
    }
    targets.forEach((t, i) => (targets[i] = Math.max(-1, Math.min(1, t))))

    let changed = false
    const next = this.keyboardAxes.map((cur, i) => {
      const target = targets[i]
      // Snap straight to zero when the key is released: ramping down adds to the perceived
      // stopping distance on top of the network round-trip. Ramp only when accelerating toward a
      // held direction, for a smooth start.
      let value: number
      if (target === 0) {
        value = 0
      } else {
        const step = keyboardRampUpPerSec * dt
        value = Math.abs(target - cur) <= step ? target : cur + Math.sign(target - cur) * step
      }
      if (value !== cur) changed = true
      return value
    })
    this.keyboardAxes = next

    if (!changed) return
    this.emitStateEvent({
      index: keyboardJoystickIndex,
      gamepad: this.buildKeyboardGamepad(),
      calibratedState: this.buildCalibratedState(next, [], JoystickModel.VirtualKeyboard),
    })
  }

  /**
   * Stop polling for gamepad events
   */
  stop(): void {
    if (this.animationFrameId !== null) {
      cancelAnimationFrame(this.animationFrameId)
      this.animationFrameId = null
    }
  }
}

export const joystickManager = JoystickManager.self()
